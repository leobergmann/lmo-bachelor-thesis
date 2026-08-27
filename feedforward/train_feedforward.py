#############################################
# Current Experiment Configuration:
# - Regression (Feedforward Network)
# - Optimizer: Choice of Scion, SGD, or AdamW
# - lr: base_lr = 0.1 (Scion), 0.0241 (SGD), 0.00985 (AdamW)
#############################################

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import uuid
import math
from math import ceil
import warnings

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from torch import nn
import torch.nn.functional as F
import logging
import torch._logging
import random
import numpy as np

# Seed and determinism
torch.use_deterministic_algorithms(True, warn_only=True)
torch._logging.set_logs(inductor=logging.ERROR, dynamo=logging.ERROR)

BASE_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


training_history = []
validation_losses = []

if len(sys.argv) > 1:
    experiment_name = sys.argv[1]
else:
    experiment_name = "standard_run_" + str(uuid.uuid4())

try:
    with open(sys.argv[0]) as f:
        code = f.read()
except FileNotFoundError:
    code = "Code reading failed (interactive mode)."


#############################################
#              Scion Optimizer              #
#############################################

@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


@torch.compile
def zeropower_via_newtonschulz6_cesista(G):
    assert len(G.shape) == 2
    abc_cesista = [
        (3.9947, -8.6494, 4.7603),
        (3.9943, -8.6482, 4.7597),
        (3.9702, -8.5041, 4.6666),
        (3.6505, -7.0837, 3.7913),
        (2.6632, -3.9414, 2.0658),
        (2.2803, -2.2275, 0.9689),
    ]
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    X = X / (X.norm() + 1e-7)
    for (a, b, c) in abc_cesista:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Norm(object):
    def lmo(self, g):
        raise NotImplementedError

    def init(self, w):
        raise NotImplementedError


class BiasRMS(Norm):
    def lmo(self, g):
        eps = 1e-8
        rms_values = torch.sqrt(torch.mean(g ** 2, dim=0, keepdim=True))
        g = g / (rms_values + eps)
        return g

    def calculate_norm(self, p):
        rms_values = torch.sqrt(torch.mean(p ** 2, dim=0, keepdim=True))
        return rms_values

    def init(self, g):
        return torch.nn.init.zeros_(g)


class SpectralConv(Norm):
    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        out_channels, in_channels, k, _ = g.shape
        g *= (out_channels / in_channels)**0.5 / (k ** 2)
        return g

    def calculate_norm(self, p):
        norm = torch.linalg.norm(p.reshape(len(p), -1), ord=2)
        out_channels, in_channels, k, _ = p.shape
        norm /= (out_channels / in_channels)**0.5 / (k ** 2)
        return norm

    def init(self, w):
        w_fp = w.data.double()
        out_channels, in_channels, k, _ = w_fp.shape
        for kx in range(k):
            for ky in range(k):
                torch.nn.init.orthogonal_(w_fp[:, :, kx, ky])

        w_fp.mul_((out_channels / in_channels)**0.5 / (k ** 2))
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class Spectral(Norm):
    def __init__(self, max=False, normalized=True, steps=5):
        self.max = max
        self.steps = steps
        self.normalized = normalized

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g, steps=self.steps)
        d_out, d_in = g.shape

        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1, scale)
        g *= scale
        return g

    def calculate_norm(self, p):
        norm = torch.linalg.norm(p, ord=2)
        d_out, d_in = p.shape
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1, scale)
        norm *= 1 / scale
        return norm

    def init(self, w):
        w_fp = w.data.double()
        torch.nn.init.orthogonal_(w_fp)
        d_out, d_in = w_fp.shape

        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1, scale)
        w_fp.mul_(scale)
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class Sign(Norm):
    def __init__(self, zero_init=False, normalized=True):
        self.zero_init = zero_init
        self.normalized = normalized

    def lmo(self, g):
        d_out, d_in = g.shape
        if self.normalized:
            return (1 / d_in) * torch.sign(g)
        else:
            return torch.sign(g)

    def calculate_norm(self, p):
        d_out, d_in = p.shape
        if self.normalized:
            return d_in * torch.max(torch.abs(p))
        else:
            return torch.max(torch.abs(p))

    def init(self, w):
        if self.zero_init:
            torch.nn.init.zeros_(w)
        else:
            d_out, d_in = w.shape
            w.data = (torch.randint(0, 2, w.shape, dtype=w.dtype, device=w.device) * 2 - 1)
            if self.normalized:
                w.data *= (1 / d_in)
        return w


class Auto(Norm):
    def lmo(self, g):
        if g.ndim in [3, 4]:
            return SpectralConv().lmo(g)
        elif g.ndim == 2:
            return Spectral().lmo(g)
        elif g.ndim in [0, 1]:
            return BiasRMS().lmo(g)

    def calculate_norm(self, p):
        if p.ndim in [3, 4]:
            return SpectralConv().calculate_norm(p)
        elif p.ndim == 2:
            return Spectral().calculate_norm(p)
        elif p.ndim in [0, 1]:
            return BiasRMS().calculate_norm(p)

    def init(self, w):
        if w.ndim in [3, 4]:
            return SpectralConv().init(w)
        elif w.ndim == 2:
            return Spectral().init(w)
        elif w.ndim in [0, 1]:
            return BiasRMS().init(w)


norm_dict = {'Sign': Sign, 'Auto': Auto}


class Scion(torch.optim.Optimizer):
    def __init__(self, params, model=None, lr=1e-3, momentum=1.0, norm: str = 'Auto', norm_kwargs: dict = None, scale=1.0, unconstrained=False):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0: raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None: norm_kwargs = {}

        self.params_vector = []
        self.grads_vector = []
        self.iter_k = 0
        self.param_to_name = {param: name for name, param in model.named_parameters()} if model is not None else {}
        defaults = dict(lr=lr, momentum=momentum, scale=scale, unconstrained=unconstrained, norm=norm, norm_kwargs=norm_kwargs)
        super().__init__(params, defaults)

    def step(self, step_epoch, iter, current_loss=None, rotate_ith_group_factor=None):
        if (iter + 1) % 2 == 0:
            self.params_vector = []
            self.grads_vector = []
            self.iter_k = 0

        global_layer_idx = 0
        total_layers = len([p for p in self.param_to_name.keys() if p.requires_grad])

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            scale = group['scale']
            unconstrained = group['unconstrained']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])

            for p in group['params']:
                name = self.param_to_name.get(p, "Unknown_Layer")

                if (iter + 1) % 2 == 0:
                    self.params_vector.append(p.data.clone())
                    self.grads_vector.append(p.grad.clone())

                g = p.grad
                if g is None:
                    global_layer_idx += 1
                    continue

                state = self.state[p]

                if momentum != 1:
                    if 'momentum_buffer' not in state.keys():
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(1 - momentum).add_(g, alpha=momentum)
                    g = buf

                if iter > 0 and iter % 2 == 0:
                    norm_params_diff = norm_backend.calculate_norm(p.data - self.params_vector[self.iter_k])
                    norm_grad_diff = torch.dot(norm_backend.lmo(p.grad - self.grads_vector[self.iter_k]).flatten().to(torch.float32), (p.grad - self.grads_vector[self.iter_k]).flatten())
                    norm_grad = torch.dot(norm_backend.lmo(p.grad).flatten().to(torch.float32), p.grad.flatten())
                    L_est = norm_grad_diff / (norm_params_diff + 1e-8)
                    param_size = p.data.size()

                    training_history.append({
                        "epoch": step_epoch,
                        "layer_index": self.iter_k,
                        "layer_name": str(name),
                        "norm_type": group['norm'],
                        "param_size": str(list(param_size)),
                        "L_estimated": L_est.item(),
                        "norm_grad": norm_grad.item(),
                        "loss": current_loss
                    })

                    if rotate_ith_group_factor is not None:
                        last_logical_step = (iter // 2) - 1
                        was_scaled_last_step = (global_layer_idx == (last_logical_step % total_layers))
                        training_history[-1]["special_ith_stepsize"] = was_scaled_last_step

                    self.iter_k += 1

                update = scale * norm_backend.lmo(g)

                if rotate_ith_group_factor is not None:
                    current_logical_step = iter // 2
                    if global_layer_idx == (current_logical_step % total_layers):
                        update *= rotate_ith_group_factor

                if not unconstrained:
                    p.data.mul_(1 - lr)
                p.data.add_(update, alpha=-lr)

                global_layer_idx += 1

    def init(self):
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            init_func = norm_backend.init
            scale = group['scale']
            for p in group['params']:
                init_func(p)
                p.data *= scale


#############################################
#            Custom Optimizers              #
#############################################

class SGD:
    """SGD with weight decay"""

    def __init__(self, params, lr=0.0241, weight_decay=0.00495, lr_decay=0.99):
        self.params = list(params)
        self.initial_lr = lr
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                p.mul_(1 - self.lr * self.weight_decay)
                p.sub_(p.grad, alpha=self.lr)
            self.lr *= self.lr_decay
        return self.params


class AdamW:
    def __init__(self, params, lr=0.00985, beta1=0.9, beta2=0.999, weight_decay=0.0904, lr_decay=1.0, eps=1e-8):
        self.params = list(params)
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.eps = eps

        self.v1 = [torch.zeros_like(p) for p in params]
        self.v2 = [torch.zeros_like(p) for p in params]
        self.step_i = 0

    def step(self):
        self.step_i += 1
        with torch.no_grad():
            for p_i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                grad = p.grad

                self.v1[p_i] = (self.beta1) * self.v1[p_i] + (1 - self.beta1) * grad
                self.v2[p_i] = (self.beta2) * self.v2[p_i] + (1 - self.beta2) * (grad ** 2)

                v1_pi_head = self.v1[p_i] / (1 - self.beta1 ** self.step_i)
                v2_pi_head = self.v2[p_i] / (1 - self.beta2 ** self.step_i)
                update = self.lr * v1_pi_head / (torch.sqrt(v2_pi_head) + self.eps)

                p.sub_(self.lr * self.weight_decay * p + update)

            self.lr *= self.lr_decay
        return self.params


#############################################
#                DataLoader                 #
#############################################

class RegressionDataLoader:
    def __init__(self, file_path="dataset_2175_kin8nm.csv", train=True, batch_size=500, device="cuda"):
        self.batch_size = batch_size
        self.train = train
        self.device = device

        data = np.loadtxt(file_path, delimiter=",", skiprows=1)

        # Ensure train and test splits are strictly reproducible
        rng = np.random.default_rng(1234)
        rng.shuffle(data)

        x_data = data[:, :-1]
        y_data = data[:, -1]

        train_fraction = 0.8
        n_train = int(x_data.shape[0] * train_fraction)

        x_train = x_data[:n_train]
        y_train = y_data[:n_train]
        x_test = x_data[n_train:]
        y_test = y_data[n_train:]

        # Calculate scaling statistics strictly based on training data
        x_mean = np.mean(x_train, axis=0, keepdims=True)
        y_mean = np.mean(y_train, keepdims=True)
        x_std = np.std(x_train, axis=0, ddof=1, keepdims=True)
        y_std = np.std(y_train, ddof=1, keepdims=True)

        if self.train:
            self.x = torch.tensor((x_train - x_mean) / x_std, dtype=torch.float32, device=self.device)
            self.y = torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32, device=self.device)
        else:
            self.x = torch.tensor((x_test - x_mean) / x_std, dtype=torch.float32, device=self.device)
            self.y = torch.tensor((y_test - y_mean) / y_std, dtype=torch.float32, device=self.device)

        self.num_samples = self.x.shape[0]

    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        if self.train:
            indices = torch.randperm(self.num_samples, device=self.device)
        else:
            indices = torch.arange(self.num_samples, device=self.device)

        for i in range(len(self)):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]

            yield self.x[batch_indices], self.y[batch_indices]


#############################################
#            Network Definition             #
#############################################

class Feedforward(nn.Module):
    def __init__(self, x_test, y_test, device, dropout_rate=0.2):
        super().__init__()
        self.device = device
        self.x_test = x_test.to(device)
        self.y_test = y_test.to(device)

        d_in = 8
        h1, h2, h3 = 256, 128, 64
        d_out = 1

        self.layers = nn.Sequential(
            nn.Linear(d_in, h1, bias=False),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(h1, h2, bias=False),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(h2, h3, bias=False),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(h3, d_out, bias=True)
        )

        self.loss_function = nn.MSELoss()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == d_out:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.requires_grad = False

        self.to(device)

    def forward(self, x):
        return self.layers(x).view(-1)

    def eval_test_error(self):
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(self.x_test)
            loss = self.loss_function(y_pred, self.y_test.view(-1))
        self.train()
        return loss.item()


def set_up_optimizer(model, base_lr=0.1, unconstrained=True, momentum=0.5, use_L1_calculated_scale=False, radius=None, scale_factor=None):
    trainable_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    L1s_experimental = [1.0] * len(trainable_params)

    if use_L1_calculated_scale:
        if radius is not None or scale_factor is not None:
            warnings.warn("radius and scale_factor are given to set_up_optimizer but will not be used, since use_L1_calculated_scale=True")

        optim_groups = []
        for i, (name, p) in enumerate(trainable_params):
            if "layers.12.weight" in name:
                norm_type = 'Sign'
                norm_kwargs = {'normalized': True}
            else:
                norm_type = 'Auto'
                norm_kwargs = {}

            optim_groups.append({
                'params': [p],
                'norm': norm_type,
                'norm_kwargs': norm_kwargs,
                'scale': 1.0 / L1s_experimental[i]
            })
    else:
        remaining_parameters = [p for n, p in model.named_parameters() if "layers.12.weight" not in n and p.requires_grad]
        output_layer_weight = [p for n, p in model.named_parameters() if "layers.12.weight" in n and p.requires_grad]

        if radius is None: radius = 1.0
        if scale_factor is None: scale_factor = 100

        optim_groups = [{
            'params': remaining_parameters,
            'norm': 'Auto',
            'norm_kwargs': {},
            'scale': radius,
        }, {
            'params': output_layer_weight,
            'norm': 'Sign',
            'norm_kwargs': {'normalized': True},
            'scale': radius * scale_factor,
        }]

    optimizer = Scion(optim_groups, model=model, lr=base_lr, momentum=momentum, unconstrained=unconstrained)
    optimizer.init()
    return optimizer


#############################################
#                 Logging                   #
#############################################

class GPUTimer:
    def __init__(self):
        self.time_seconds = 0.0
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)

    def start(self): self.starter.record()

    def stop(self):
        self.ender.record()
        torch.cuda.synchronize()
        self.time_seconds += 1e-3 * self.starter.elapsed_time(self.ender)


def save_experiment_logs(experiment_name, final_loss, code_string, history, validation_losses):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs", experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(
        code=code_string,
        final_loss=final_loss,
        run_name=experiment_name,
        history=history,
        validation_losses=validation_losses
    ), log_path)

    print(f"\nTraining completed! Data saved locally in:")
    print(os.path.abspath(log_path))


def get_leaf_layers(model):
    leaf_layers = []

    def _traverse(module, prefix=""):
        children = list(module.named_children())
        if children:
            for name, child in children:
                full_name = f"{prefix}.{name}" if prefix else name
                _traverse(child, full_name)
        else:
            if any(p.requires_grad for p in module.parameters()):
                leaf_layers.append((prefix, module))

    _traverse(model)
    return leaf_layers


#############################################
#                Training                   #
#############################################

def main(n_runs=1):
    n_epochs = 10
    batch_size = 64
    optimizer_choice = "Scion"  # feedforward_kin8nm_uS_nS_10epochs_10runs
    # optimizer_choice = "AdamW"   # feedforward_kin8nm_AdamW_nS_10epochs_10runs
    # optimizer_choice = "SGD"    # feedforward_kin8nm_SGD_nS_10epochs_10runs

    base_lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_matrix = np.zeros((n_runs, n_epochs))
    val_matrix = np.zeros((n_runs, n_epochs))
    run_times = []

    history_records = []

    print(f"Starting evaluation averaged over {n_runs} runs...")

    for run in range(n_runs):
        run_seed = BASE_SEED + run
        set_seed(run_seed)
        print(f"\n--- Run {run + 1}/{n_runs} (Seed: {run_seed}) ---")

        train_loader = RegressionDataLoader(file_path="dataset_2175_kin8nm.csv", train=True, batch_size=batch_size, device=device)
        test_loader = RegressionDataLoader(file_path="dataset_2175_kin8nm.csv", train=False, batch_size=batch_size, device=device)

        model = Feedforward(x_test=test_loader.x, y_test=test_loader.y, device=device)

        target_layers = get_leaf_layers(model)

        optim_params = [p for p in model.parameters() if p.requires_grad]
        if optimizer_choice == "Scion":
            optimizer = set_up_optimizer(model=model, base_lr=base_lr, unconstrained=True, momentum=0.5, use_L1_calculated_scale=False)
        elif optimizer_choice == "SGD":
            optimizer = SGD(optim_params, lr=0.0241)
        elif optimizer_choice == "AdamW":
            optimizer = AdamW(optim_params, lr=0.00985)
        else:
            raise ValueError("Unknown optimizer selected.")

        timer = GPUTimer()
        total_train_steps = math.ceil(n_epochs * len(train_loader))
        step = 0

        prev_weights = {}
        prev_grads = {}

        timer.start()
        for epoch in range(n_epochs):
            model.train()
            total_train_loss = 0.0

            epoch_L_estimates = {layer_name: [] for layer_name, _ in target_layers}
            epoch_grad_norms = {layer_name: [] for layer_name, _ in target_layers}

            for iter, (inputs, targets) in enumerate(train_loader):
                outputs = model(inputs)
                loss = F.mse_loss(outputs.view(-1), targets.view(-1), reduction="mean")
                loss.backward()

                total_train_loss += loss.item()

                for layer_name, layer_module in target_layers:
                    params = [p for p in layer_module.parameters() if p.requires_grad]

                    curr_grads = [p.grad.detach().clone() for p in params if p.grad is not None]
                    curr_weights = [p.detach().clone() for p in params if p.grad is not None]

                    if curr_grads:
                        grad_norm = torch.sqrt(sum(g.norm()**2 for g in curr_grads)).item()

                        if layer_name in prev_weights:
                            grad_diff = torch.sqrt(sum((g - pg).norm()**2 for g, pg in zip(curr_grads, prev_grads[layer_name]))).item()
                            weight_diff = torch.sqrt(sum((w - pw).norm()**2 for w, pw in zip(curr_weights, prev_weights[layer_name]))).item()

                            L_est = grad_diff / (weight_diff + 1e-8)
                            epoch_L_estimates[layer_name].append(L_est)

                        epoch_grad_norms[layer_name].append(grad_norm)

                        prev_weights[layer_name] = curr_weights
                        prev_grads[layer_name] = curr_grads

                if isinstance(optimizer, Scion):
                    step_epoch = step / len(train_loader)
                    optimizer.step(step_epoch=step_epoch, iter=iter, current_loss=loss.item())
                else:
                    optimizer.step()

                model.zero_grad(set_to_none=True)
                step += 1
                if step >= total_train_steps:
                    break

            epoch_train_loss = total_train_loss / len(train_loader)
            epoch_val_loss = model.eval_test_error()

            train_matrix[run, epoch] = epoch_train_loss
            val_matrix[run, epoch] = epoch_val_loss

            for group_idx, (layer_name, layer_module) in enumerate(target_layers):
                params = [p for p in layer_module.parameters() if p.requires_grad]

                avg_L = np.mean(epoch_L_estimates[layer_name]) if epoch_L_estimates.get(layer_name) else 1.0
                avg_grad_norm = np.mean(epoch_grad_norms[layer_name]) if epoch_grad_norms.get(layer_name) else 0.0

                readable_name = f"{layer_name} ({layer_module.__class__.__name__})"
                param_sizes = ", ".join([str(tuple(p.shape)) for p in params])

                history_records.append({
                    "epoch": epoch,
                    "layer_index": group_idx,
                    "layer_name": readable_name,
                    "param_size": param_sizes,
                    "norm_grad": avg_grad_norm,
                    "L_estimated": avg_L,
                    "loss": epoch_train_loss
                })

        timer.stop()
        run_times.append(timer.time_seconds)
        print(f"Run {run + 1} finished in {timer.time_seconds:.2f}s | Final Val Loss: {val_matrix[run, -1]:.4f}")

    mean_val = np.mean(val_matrix, axis=0)
    std_val = np.std(val_matrix, axis=0)

    return {
        "final_loss": float(mean_val[-1]),
        "history": history_records,
        "validation_losses": mean_val.tolist(),
        "val_std": std_val.tolist()
    }


if __name__ == "__main__":
    results = main()

    save_experiment_logs(
        experiment_name=experiment_name,
        final_loss=results["final_loss"],
        code_string=code,
        history=results["history"],
        validation_losses=results["validation_losses"]
    )