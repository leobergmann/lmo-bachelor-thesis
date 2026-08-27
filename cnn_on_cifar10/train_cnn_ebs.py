#############################################
# Setting des aktuellen Experiments:
# - Cnn
# - uS: unScion
# - cif: CIFAR10 (not mnist: MNIST )(not cif100) 
# - nS: ohne Scheduler
# - lr: base_lr: learning_rate = 0.1
# - exs: explizite scales (use_L1_calculated_scale=False):
#       + radius = 1.0
#       + scale_factor = 100)
# - (not calcs: calculated scales based on experiments)
# - mom: momentum = 0.5
# - epoch: 80 epochs
#############################################


#############################################
#                  Setup                    #
#############################################

import os
# make it deterministic
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
with open(sys.argv[0]) as f:
    code = f.read()
# NEW (NEW0: same with "run_name" instead of "experiment_name")
if len(sys.argv) > 1:
    experiment_name = sys.argv[1]
else:
    experiment_name = "standard_run_" + str(uuid.uuid4())
import uuid
import math
from math import ceil
import warnings

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from torch import nn
from torch.optim.swa_utils import update_bn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import logging
import torch._logging
# Just for seeding numpy an random. Other libraries may use numpy and random
import random
import numpy as np


torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 2. Der absolute Seed-Block für alle Ebenen
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)                # PyTorch CPU
torch.cuda.manual_seed(SEED)           # PyTorch GPU
torch.cuda.manual_seed_all(SEED)       # PyTorch GPU (Multi-GPU)

# (Optional: Verbietet PyTorch Funktionen, die sich nicht deterministisch verhalten können)
torch.use_deterministic_algorithms(True, warn_only=True) 

torch._logging.set_logs(inductor=logging.ERROR, dynamo=logging.ERROR)


#############################################
# NEW Lokales Logging
training_history = []
validation_accs = []
#############################################


#############################################
#               Scion optimizer             #
#############################################

@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    """
    From: https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101724_DistributedMuon/22d24867-eb5a-4fcc-ae2c-263d0277dfd1.txt
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X


def zeroth_power_via_svd(G):
   U, S, V = G.svd()
   return U @ V.T

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
        # new: der Code kann eigentlich hier schon stehen
        out_channels, in_channels, k, _ = w_fp.shape
        # old: k = w.data.size(2)
        for kx in range(k):
            for ky in range(k):
                torch.nn.init.orthogonal_(w_fp[:,:,kx,ky])
        
        # old: out_channels, in_channels, k, _ = w_fp.shape
        w_fp.mul_((out_channels / in_channels)**0.5 / (k ** 2))
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class Spectral(Norm):
    def __init__(self, max=False, normalized=True, steps=5):
        self.max = max
        self.steps = steps
        self.normalized = normalized

    def lmo(self, g):
        # old: g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        # new: (reshape und view sollten nicht noetig sein, da man bereits eine 2D Matrix hat)
        g = zeropower_via_newtonschulz5(g, steps=self.steps)
        d_out, d_in = g.shape
        
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1,scale)
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
            scale = max(1,scale)
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
            scale = max(1,scale)
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
            return (1/d_in)*torch.sign(g)    
        else:
            return torch.sign(g)

    def calculate_norm(self, p):
        d_out, d_in = p.shape
        if self.normalized:
            return d_in*torch.max(torch.abs(p))    
        else:
            return torch.max(torch.abs(p))

    def init(self, w):
        if self.zero_init:
            torch.nn.init.zeros_(w)
        else:
            # Generate -1/fan_in or 1/fan_in uniformly at random
            d_out, d_in = w.shape
            w.data = (torch.randint(0, 2, w.shape, dtype=w.dtype, device=w.device) * 2 - 1)
            if self.normalized:
                w.data *= (1/d_in)
        return w


class Auto(Norm):
    def lmo(self, g):
        if g.ndim in [3,4]:
            return SpectralConv().lmo(g)
        elif g.ndim == 2:
            return Spectral().lmo(g)
        elif g.ndim in [0,1]:
            return BiasRMS().lmo(g)

    def calculate_norm(self, p):
        if p.ndim in [3,4]:
            return SpectralConv().calculate_norm(p)
        elif p.ndim == 2:
            return Spectral().calculate_norm(p)
        elif p.ndim in [0,1]:
            return BiasRMS().calculate_norm(p)

    def init(self, w):
        if w.ndim in [3,4]:
            return SpectralConv().init(w)
        elif w.ndim == 2:
            return Spectral().init(w)
        elif w.ndim in [0,1]:
            return BiasRMS().init(w)


norm_dict = {
    'Sign': Sign,
    'Auto': Auto,
}


class Scion(torch.optim.Optimizer):
    """Scion optimizer implementation.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        momentum (float, optional): One minus the traditional momentum factor. For example,
            a traditional momentum of 0.9 would be specified as momentum=0.1 here (default: 1.0)
        norm (str, optional): Choice of norm for gradient projection ('Auto', 'SpectralConv', 
            'ColNorm', 'RowNorm', 'BiasRMS', 'Spectral', or 'Sign') (default: 'Auto')
        norm_kwargs (dict, optional): Additional arguments for the norm projection (default: None)
        scale (float, optional): Scale factor for updates (default: 1.0)
        unconstrained (bool, optional): Whether to use unconstrained updates (default: False)
    """
    def __init__(self, params, model=None, lr=1e-3, momentum=1.0, norm: str='Auto', norm_kwargs: dict=None, scale=1.0, unconstrained=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = {}
        self.params_vector = []  
        self.grads_vector = []  
        self.iter_k = 0
        if model is not None:
            self.param_to_name = {param: name for name, param in model.named_parameters()}
        else:
            self.param_to_name = {}
        defaults = dict(lr=lr, momentum=momentum, scale=scale, unconstrained=unconstrained, norm=norm, norm_kwargs=norm_kwargs)
        super().__init__(params, defaults)

    def step(self, step_epoch, iter, current_loss=None, rotate_ith_group_factor=None):
        if (iter+1) % 2 == 0:
            self.params_vector = []
            self.grads_vector = []
            self.iter_k = 0
        
        

        global_layer_idx = 0
        total_layers = 14

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            scale = group['scale']
            unconstrained = group['unconstrained']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            for p in group['params']:
                
                name = self.param_to_name.get(p, "Unknown_Layer")

                if (iter+1) % 2 == 0:
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
                    buf.mul_(1-momentum).add_(g, alpha=momentum)
                    g = buf

                if iter > 0 and iter % 2 == 0:
                    norm_params_diff = norm_backend.calculate_norm(p.data - self.params_vector[self.iter_k])
                    norm_grad_diff = torch.dot(norm_backend.lmo(p.grad - self.grads_vector[self.iter_k]).flatten().to(torch.float32),  (p.grad - self.grads_vector[self.iter_k]).flatten()) 
                    norm_grad = torch.dot(norm_backend.lmo(p.grad).flatten().to(torch.float32), p.grad.flatten())
                    L_est = norm_grad_diff / norm_params_diff
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
                        
                    print(f'step:{step_epoch:.4f} ({self.iter_k}) L_estimated: {L_est.item():.4f} norm_grad: {norm_grad.item():.4f}')  
                    self.iter_k += 1

                update = scale * norm_backend.lmo(g)

                if rotate_ith_group_factor is not None:
                    current_logical_step = iter // 2
                    if global_layer_idx == (current_logical_step % total_layers):
                        update *= rotate_ith_group_factor

                if not unconstrained:
                    p.data.mul_(1-lr)
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
#                DataLoader                 #
#############################################


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)

    c = images.size(1)

    images_out = torch.empty((len(images), c, crop_size, crop_size), device=images.device, dtype=images.dtype)
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), c, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class ImageDataLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None, loader=None, IMAGE_MEAN=None, IMAGE_STD=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        
        if path == "cifar10":
            if loader is None: loader = torchvision.datasets.CIFAR10
            if IMAGE_MEAN is None: IMAGE_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
            if IMAGE_STD is None: IMAGE_STD = torch.tensor((0.2470, 0.2435, 0.2616))
        elif path == "cifar100": 
            if loader is None : loader = torchvision.datasets.CIFAR100
            if IMAGE_MEAN is None : IMAGE_MEAN = torch.tensor((0.5071, 0.4867, 0.4408))
            if IMAGE_STD is None : IMAGE_STD = torch.tensor((0.2675, 0.2565, 0.2761))
        elif path == "mnist":
            if loader is None: loader = torchvision.datasets.MNIST
            if IMAGE_MEAN is None: IMAGE_MEAN = torch.tensor((0.1307,))
            if IMAGE_STD is None: IMAGE_STD  = torch.tensor((0.3081,))

        if not os.path.exists(data_path):
            dset = loader(path, download=True, train=train)
            
            images = torch.as_tensor(dset.data).clone().detach()
            labels = torch.as_tensor(dset.targets).clone().detach()
            
            if path == "cifar10" or path == "cifar100":
                images = images.permute(0, 3, 1, 2)
            elif path == "mnist":
                images = images.unsqueeze(1)
                
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device("cuda"))
        self.images, self.labels, self.classes = data["images"], data["labels"], data["classes"]
        self.images = (self.images.float() / 255).to(memory_format=torch.channels_last)
        self.normalize = T.Normalize(IMAGE_MEAN, IMAGE_STD)
        self.proc_images = {} # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate"], "Unrecognized key: %s" % k

        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):

        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,)*4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get("flip", False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Definition             #
#############################################

# note the use of low BatchNorm stats momentum
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

class ImageNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_blocks=3):
        super().__init__()

        if num_blocks not in [1, 2, 3]:
            raise ValueError("num_blocks must be 1, 2 or 3")
        
        widths = [64, 256, 256]

        whiten_kernel_size = 2
        whiten_width = 2 * n_channels * whiten_kernel_size**2

        self.whiten = nn.Conv2d(n_channels, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False

        current_in_channels = whiten_width
        self.layers = [nn.GELU()]

        for i in range(num_blocks):
            current_out_channels = widths[i]
            self.layers.append(ConvGroup(current_in_channels, current_out_channels))
            current_in_channels = current_out_channels

        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        self.layers = nn.Sequential(*self.layers)
        
        self.head = nn.Linear(current_in_channels, n_classes, bias=False)
    
        for mod in self.modules():
            mod.float()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w) / torch.sqrt(eigenvalues.view(-1,1,1,1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b)
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ""
    for col in columns_list:
        print_string += "|  %s  " % col
    print_string += "|"
    if is_head:
        print("-"*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-"*len(print_string))

logging_columns_list = ["epoch", "train_acc", "val_acc", "tta_val_acc", "time_seconds"]
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = "{:0.4f}".format(var)
        else:
            assert var is None
            res = ""
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#               Evaluation                 #
############################################

def infer(model, loader, image_set, tta_level=0):

    # Test-time augmentation strategy (for tta_level=2):
    # 1. Flip/mirror the image left-to-right (50% of the time).
    # 2. Translate the image by one pixel either up-and-left or down-and-right (50% of the time,
    #    i.e. both happen 25% of the time).
    #
    # This creates 6 views per image (left/right times the two translations and no-translation),
    # which we evaluate and then weight according to the given probabilities.

    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        if image_set == "mnist":
            return net(inputs).clone()
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, "reflect")

        h, w = inputs.shape[2:]

        inputs_translate_list = [
            padded_inputs[:, :, 0:h, 0:h],
            padded_inputs[:, :, 2:h+2, 2:h+2],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    
    # Reset so that loader acts like on new train-data
    loader.epoch = 0
    loader.proc_images = {}

    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, image_set, tta_level=0):
    logits = infer(model, loader, image_set, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#                Training                  #
############################################

def set_up_optimizer(model, base_lr=0.5, unconstrained=True, momentum=1.0, use_L1_calculated_scale=False, radius=None, scale_factor=None):
    """If use_L1_calculated_scale=True, we three optim groups (conv, norm, output) radius and scale=1/L^1, 
    where L^1 is an approximation of L^1 for each specific group (L^0 is assumed to be 0).
    if noise should be considered (Zeta in assumption 4), multiply your base_lr with (1-Zeta)/(1+Zeta)
    """
    # experimental results cnn_cif_uS_nS_exs_80epochs_0p1_lr:
    L1s_experimental = [14.70, 13.95, 13.48, 13.59, 13.86, 16.13, 18.20, 17.46, 17.02, 16.71, 17.32, 20.07, 20.56, 0.20]
    if use_L1_calculated_scale:
        if radius is not None or scale_factor is not None:
            warnings.warn("radius and scale_factor are given to set_up_optimizer but will not be used, since use_L1_calculated_scale=True")
        
        # Alle trainierbaren Parameter in der Original-Reihenfolge des Netzwerks abrufen
        trainable_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        
        if len(trainable_params) != len(L1s_experimental):
            raise ValueError(f"Mismatch: Model has {len(trainable_params)} trainable parameters, but L1s_experimental has {len(L1s_experimental)} values.")
        
        optim_groups = []
        
        for i, (name, p) in enumerate(trainable_params):
            if "head" in name:
                norm_type = 'Sign'
                norm_kwargs = {'normalized': True}
            else:
                norm_type = 'Auto'
                norm_kwargs = {}
                
            # Erstelle für jeden der 14 Parameter eine exklusive Parameter-Gruppe
            optim_groups.append({
                'params': [p],  # Nur dieser eine Parameter
                'norm': norm_type,
                'norm_kwargs': norm_kwargs,
                'scale': 1.0 / L1s_experimental[i]  # Der exakte, gemessene Kehrwert
            })

    else:
        # Create optimizers and schedulers
        filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
        norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
        remaining_parameters = filter_params + [model.whiten.bias] + norm_biases
        output_layer = [model.head.weight]
        if radius is None:
            radius = 1.0
        if scale_factor is None:
            scale_factor = 100
        optim_groups = [{
            'params': remaining_parameters,
            'norm': 'Auto',
            'norm_kwargs': {},
            'scale': radius,
        }, {
            'params': output_layer,
            'norm': 'Sign',
            'norm_kwargs': {'normalized': True},
            'scale': radius*scale_factor,
        }]
    
    
    optimizer1 = Scion(optim_groups, model=model, lr=base_lr, momentum=momentum, unconstrained=unconstrained)
    optimizer1.init()
    return optimizer1


class GPUTimer:
    def __init__(self):
        self.time_seconds = 0.0
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.starter.record()

    def stop(self):
        self.ender.record()
        torch.cuda.synchronize()
        self.time_seconds += 1e-3 * self.starter.elapsed_time(self.ender)

def save_experiment_logs(experiment_name, final_acc_list, code_string, history, validation_accs):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs", experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, "log.pt")
    
    # very unfortunate naming of "accs" (just the final validation-accuracy)
    torch.save(dict(
        code=code_string,
        accs=final_acc_list,
        run_name=experiment_name,
        history=history,
        validation_accs=validation_accs
    ), log_path)

    print(f"\n Training abgeschlossen! Daten lokal gespeichert in:")
    print(os.path.abspath(log_path))


def main():
    # --- 1. Configuration & Setup ---
    n_epochs = 160
    batch_size = 500
    image_set = "cifar10"
    
    base_lr = 0.1
    unconstrained = True
    momentum = 0.9
    
    use_L1_calculated_scale = False


    if image_set == "cifar10" or image_set == "cifar100":
        n_channels = 3
        flip_images = True
    elif image_set == "mnist":
        n_channels = 1
        # a fliped 2 can look like a five
        flip_images = False

    n_classes = 100 if image_set == "cifar100" else 10

    test_loader = ImageDataLoader(image_set, train=False, batch_size=batch_size)
    train_loader = ImageDataLoader(image_set, train=True, batch_size=batch_size, aug=dict(flip=flip_images, translate=2))
    
    total_train_steps = math.ceil(n_epochs * len(train_loader))
    total_epochs = math.ceil(total_train_steps / len(train_loader))

    model = ImageNet(n_channels=n_channels, n_classes=n_classes).cuda().to(memory_format=torch.channels_last)
    # 20 is just a random number, I found it faster if it is not used for 8 epochs
    if n_epochs > 20:
        model.compile(mode="max-autotune")
    else:
        model.compile()
    model.reset()

    optimizer1 = set_up_optimizer(model=model, base_lr=base_lr, unconstrained=unconstrained, momentum=momentum, use_L1_calculated_scale=use_L1_calculated_scale)
    optimizers = [optimizer1]

    def linear_decay(step):
        return max(0.0, 1.0 - step / total_train_steps)
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=linear_decay) for opt in optimizers]

    timer = GPUTimer()

    # --- 3. Whitening Initialisation ---
    timer.start()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    timer.stop()

    # --- 4. Training ---
    step = 0
    for epoch in range(total_epochs):
        
        timer.start()
        model.train()
        
        for iter, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum")
            loss.backward()
            
            for opt, sched in zip(optimizers, schedulers):
                step_epoch = step / len(train_loader)
                opt.step(step_epoch=step_epoch, iter=iter, current_loss=loss.item())
                # sched.step() # Auskommentiert gelassen, wie in deinem Original
            
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
                
        timer.stop()

        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        
        # calibrate BatchNorm running statistics
        train_loader.epoch = 0
        train_loader.proc_images = {}
        update_bn(train_loader, model, device=torch.device("cuda"))

        val_acc = evaluate(model, test_loader, image_set=image_set, tta_level=0)
        validation_accs.append(val_acc)
        
        # locals() braucht 'time_seconds', um es an print_training_details zu übergeben
        time_seconds = timer.time_seconds 
        print_training_details(locals(), is_final_entry=False)


    # --- 5. Test-Time Augmentation (TTA) Evaluierung ---
    timer.start()
    tta_val_acc = evaluate(model, test_loader, image_set=image_set, tta_level=2)
    timer.stop()
    
    epoch = "eval"
    time_seconds = timer.time_seconds
    print_training_details(locals(), is_final_entry=True)

    return tta_val_acc


if __name__ == "__main__":
    print_columns(logging_columns_list, is_head=True)
    
    final_acc_list = main()

    save_experiment_logs(experiment_name, final_acc_list, code, training_history, validation_accs)
