# Layer-Wise Smoothness Evaluation & Optimizer Benchmarks (unScion / Gluon)

This repository contains code to evaluate **layer-wise trajectory smoothness** $(L^0, L^1)$, explore **generalized Newton-Schulz iterations**, and benchmark optimizers (**unScion**, **AdamW**, **SGD**) across different network architectures and datasets.

The codebase extends the original work on **Gluon / Scion** by Riabinin et al., **Muon** by Keller Jordan et al., and polynomial coefficient optimization by Cesista et al.

---

## Features & Extensions

* **Architectures & Datasets**:
  * **NanoGPT** trained on **FineWeb-10B**.
  * **CNNs** evaluated on **CIFAR-10**, **CIFAR-100**, and **MNIST**.
  * **Feedforward Neural Networks (FNN)** with Dropout and Batch Normalization trained on the robotics regression dataset **kin8nm**.
* **Optimizer Comparisons & Tuning**:
  * Full integration and Hyperparameter Optimization (using **Optuna**) for **AdamW** and **SGD**.
  * Support for individual, per-layer step sizes and scaling factors ($t_i$) in unScion.
* **Generalized Newton-Schulz Iteration**:
  * Optimization of high-order polynomials for matrix orthogonalization based on the approach by **Cesista et al.**
  * Learning rate scheduling and increased sample density for polynomial fitting in $[0, 1]$.
  * Comparative evaluation against standard Newton-Schulz (Jordan et al.) and the *Modula* library.
* **Empirical & Theoretical Smoothness Analysis**:
  * Layer-wise $(L^0, L^1)$-smoothness estimation over varying training durations (8 to 160 epochs) and learning rates.
  * Practical validation of theoretically derived step sizes $t_i = \frac{1}{12 L_i^1}$.

---

## Dataset Preparation

To download and cache the **FineWeb-10B** dataset for NanoGPT:

```bash
python data/cached_fineweb10B.py 8