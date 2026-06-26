# ══════════════════════════════════════════════════════════════════════════════
# CNN FROM SCRATCH  ·  NumPy / CuPy backend  (no torch.nn layers)
# ══════════════════════════════════════════════════════════════════════════════

import os

# ── Add CUDA DLL paths before importing CuPy ──────────────────────────────────
_cuda_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64",
]

for _p in _cuda_paths:
    if os.path.exists(_p):
        os.add_dll_directory(_p)

# ── Backend ───────────────────────────────────────────────────────────────────
try:
    import cupy as xp
    GPU = True
    print("Backend: GPU (CuPy)")
except ImportError:
    import numpy as xp
    GPU = False
    print("Backend: CPU (NumPy)")

import numpy as np_cpu
import math
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
)

# ── GPU helpers ───────────────────────────────────────────────────────────────

def to_xp(a, dtype=np_cpu.float32):
    a = np_cpu.asarray(a, dtype=dtype)
    return xp.asarray(a) if GPU else a

def to_numpy(a):
    if isinstance(a, np_cpu.ndarray): return a
    return xp.asnumpy(a) if GPU else np_cpu.asarray(a)

def im2col(x, kH, kW, stride=1, pad=0):
    N, C, H, W = x.shape
    oH = (H + 2*pad - kH) // stride + 1
    oW = (W + 2*pad - kW) // stride + 1

    if pad > 0:
        x = xp.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')

    # Build index arrays — all on GPU, no Python loops
    i0 = xp.repeat(xp.arange(kH), kW)                          # (kH*kW,)
    i0 = xp.tile(i0, C)                                         # (C*kH*kW,)
    i1 = stride * xp.repeat(xp.arange(oH), oW)                 # (oH*oW,)

    j0 = xp.tile(xp.arange(kW), kH)                            # (kH*kW,)
    j0 = xp.tile(j0, C)                                         # (C*kH*kW,)
    j1 = stride * xp.tile(xp.arange(oW), oH)                   # (oH*oW,)

    c_idx = xp.repeat(xp.arange(C), kH*kW)                     # (C*kH*kW,)

    # Single gather operation — one GPU kernel, not 9
    i = (i0.reshape(-1,1) + i1.reshape(1,-1))                   # (C*kH*kW, oH*oW)
    j = (j0.reshape(-1,1) + j1.reshape(1,-1))                   # (C*kH*kW, oH*oW)
    c = c_idx.reshape(-1,1)                                      # (C*kH*kW, 1)

    col = x[:, c, i, j]                                         # (N, C*kH*kW, oH*oW)
    col = col.transpose(0, 2, 1)                                 # (N, oH*oW, C*kH*kW)
    return col, oH, oW

def col2im(dcol, x_shape, kH, kW, stride=1, pad=0):
    """(N, oH*oW, C*kH*kW) → (N,C,H,W)  — inverse of im2col."""
    N, C, H, W = x_shape
    oH = (H + 2*pad - kH) // stride + 1
    oW = (W + 2*pad - kW) // stride + 1
    dcol = dcol.reshape(N, oH, oW, C, kH, kW).transpose(0, 3, 4, 5, 1, 2)
    dx = xp.zeros((N, C, H+2*pad, W+2*pad), dtype=dcol.dtype)
    for i in range(kH):
        for j in range(kW):
            dx[:, :, i:i+stride*oH:stride, j:j+stride*oW:stride] += dcol[:, :, i, j, :, :]
    return dx[:, :, pad:pad+H, pad:pad+W] if pad > 0 else dx

# ══════════════════════════════════════════════════════════════════════════════
# Layer base
# ══════════════════════════════════════════════════════════════════════════════

class Layer:
    def __init__(self):
        self.params   = {}
        self.grads    = {}
        self.training = True

    def forward(self, x):  raise NotImplementedError
    def backward(self, d): raise NotImplementedError
    def train_mode(self): self.training = True
    def eval_mode(self):  self.training = False

    def param_grad_pairs(self):
        return [(self.params, self.grads, k) for k in self.params]

# ══════════════════════════════════════════════════════════════════════════════
# Layers
# ══════════════════════════════════════════════════════════════════════════════

class Conv2d(Layer):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=0):
        super().__init__()
        self.out_ch, self.k = out_ch, k
        self.stride, self.padding = stride, padding
        std = math.sqrt(2.0 / (in_ch * k * k))
        W   = xp.random.randn(out_ch, in_ch*k*k).astype(xp.float32) * std
        self.params = {'W': W,                  'b': xp.zeros(out_ch, xp.float32)}
        self.grads  = {'W': xp.zeros_like(W),   'b': xp.zeros(out_ch, xp.float32)}
        self.cache  = None

    def forward(self, x):
        col, oH, oW = im2col(x, self.k, self.k, self.stride, self.padding)
        N = x.shape[0]
        out = (col @ self.params['W'].T + self.params['b']) \
                .transpose(0, 2, 1).reshape(N, self.out_ch, oH, oW)
        self.cache = (x, col)
        return out

    def backward(self, dout):
        x, col = self.cache
        N, C_out, oH, oW = dout.shape
        dr = dout.reshape(N, C_out, -1).transpose(0, 2, 1)
        self.grads['W'] = (dr.transpose(0, 2, 1) @ col).sum(0)
        self.grads['b'] = dr.sum(axis=(0, 1))
        dcol = dr @ self.params['W']
        return col2im(dcol, x.shape, self.k, self.k, self.stride, self.padding)


class BatchNorm2d(Layer):
    def __init__(self, C, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.mom = eps, momentum
        self.params = {'gamma': xp.ones(C,  xp.float32),
                       'beta':  xp.zeros(C, xp.float32)}
        self.grads  = {'gamma': xp.zeros(C, xp.float32),
                       'beta':  xp.zeros(C, xp.float32)}
        self.running_mean = xp.zeros(C, xp.float32)
        self.running_var  = xp.ones(C,  xp.float32)
        self.cache = None

    def forward(self, x):
        g = self.params['gamma'].reshape(1,-1,1,1)
        b = self.params['beta'].reshape(1,-1,1,1)
        if self.training:
            mu  = x.mean(axis=(0,2,3), keepdims=True)
            var = x.var(axis=(0,2,3),  keepdims=True)
            xh  = (x - mu) / xp.sqrt(var + self.eps)
            self.running_mean = (1-self.mom)*self.running_mean + self.mom*mu.ravel()
            self.running_var  = (1-self.mom)*self.running_var  + self.mom*var.ravel()
            self.cache = (x, xh, mu, var)
        else:
            mu  = self.running_mean.reshape(1,-1,1,1)
            var = self.running_var.reshape(1,-1,1,1)
            xh  = (x - mu) / xp.sqrt(var + self.eps)
        return g * xh + b

    def backward(self, dout):
        x, xh, mu, var = self.cache
        N, C, H, W = x.shape
        M  = N * H * W
        g  = self.params['gamma'].reshape(1,-1,1,1)
        self.grads['gamma'] = (dout * xh).sum(axis=(0,2,3))
        self.grads['beta']  =  dout.sum(axis=(0,2,3))
        dxh  = dout * g
        std  = xp.sqrt(var + self.eps)
        dvar = (dxh * (x-mu) * -0.5 * (var+self.eps)**-1.5).sum(axis=(0,2,3), keepdims=True)
        dmu  = (dxh / -std).sum(axis=(0,2,3), keepdims=True) \
             + dvar * (-2*(x-mu)).sum(axis=(0,2,3), keepdims=True) / M
        return dxh/std + dvar*2*(x-mu)/M + dmu/M


class ReLU(Layer):
    def __init__(self): super().__init__(); self.cache = None
    def forward(self, x):  self.cache = x;  return xp.maximum(0, x)
    def backward(self, d): return d * (self.cache > 0)


class AvgPool2d(Layer):
    def __init__(self, k, stride=None):
        super().__init__()
        self.k      = k
        self.stride = stride if stride else k
        self.cache  = None

    def forward(self, x):
        N, C, H, W = x.shape
        oH = (H - self.k) // self.stride + 1
        oW = (W - self.k) // self.stride + 1
        col, _, _ = im2col(x, self.k, self.k, self.stride, 0)
        col = col.reshape(N, oH*oW, C, self.k*self.k)
        out = col.mean(axis=3).transpose(0,2,1).reshape(N, C, oH, oW)
        self.cache = (x.shape, oH, oW)
        return out

    def backward(self, dout):
        x_shape, oH, oW = self.cache
        N, C, H, W = x_shape
        df   = dout.reshape(N, C, -1).transpose(0, 2, 1)   # (N, oH*oW, C)
        dcol = xp.repeat(df[:,:,:,None], self.k*self.k, axis=3) / (self.k*self.k)
        return col2im(dcol.reshape(N, oH*oW, C*self.k*self.k),
                      x_shape, self.k, self.k, self.stride, 0)

class MaxPool2d(Layer):
    def __init__(self, k, stride=None):
        super().__init__()
        self.k      = k
        self.stride = stride if stride else k
        self.cache  = None

    def forward(self, x):
        N, C, H, W = x.shape
        oH = (H - self.k) // self.stride + 1
        oW = (W - self.k) // self.stride + 1
        col, _, _ = im2col(x, self.k, self.k, self.stride, 0)
        col = col.reshape(N, oH*oW, C, self.k*self.k)
        idx = xp.argmax(col, axis=3)
        out = col.max(axis=3).transpose(0,2,1).reshape(N, C, oH, oW)
        self.cache = (x.shape, col, idx, oH, oW)
        return out

    def backward(self, dout):
        x_shape, col, idx, oH, oW = self.cache
        N, C, H, W = x_shape
        df   = dout.reshape(N, C, -1).transpose(0, 2, 1)
        dcol = xp.zeros_like(col)
        ni = xp.arange(N)[:,None,None]
        hi = xp.arange(oH*oW)[None,:,None]
        ci = xp.arange(C)[None,None,:]
        dcol[ni, hi, ci, idx] = df
        return col2im(dcol.reshape(N, oH*oW, C*self.k*self.k),
                      x_shape, self.k, self.k, self.stride, 0)


class AdaptiveMaxPool2d(Layer):
    def __init__(self, out_size):
        super().__init__()
        self.oH, self.oW = (out_size, out_size) if isinstance(out_size, int) else out_size
        self.cache = None

    @staticmethod
    def _bins(in_sz, out_sz):
        return [(int(math.floor(i*in_sz/out_sz)),
                 int(math.ceil((i+1)*in_sz/out_sz))) for i in range(out_sz)]

    def forward(self, x):
        N, C, H, W = x.shape
        hb = self._bins(H, self.oH)
        wb = self._bins(W, self.oW)
        out       = xp.zeros((N, C, self.oH, self.oW), dtype=x.dtype)
        max_h_idx = xp.zeros((N, C, self.oH, self.oW), dtype=xp.int32)
        max_w_idx = xp.zeros((N, C, self.oH, self.oW), dtype=xp.int32)

        for i, (h0, h1) in enumerate(hb):
            for j, (w0, w1) in enumerate(wb):
                patch = x[:, :, h0:h1, w0:w1]          # (N, C, ph, pw)
                ph, pw = h1-h0, w1-w0
                flat  = patch.reshape(N, C, ph*pw)       # (N, C, ph*pw)
                idx   = xp.argmax(flat, axis=2)          # (N, C)
                ni = xp.arange(N)[:, None]
                ci = xp.arange(C)[None, :]
                out[:, :, i, j]       = flat[ni, ci, idx]
                max_h_idx[:, :, i, j] = h0 + idx // pw  # absolute h position
                max_w_idx[:, :, i, j] = w0 + idx % pw   # absolute w position

        self.cache = (x.shape, max_h_idx, max_w_idx)
        return out

    def backward(self, dout):
        x_shape, max_h_idx, max_w_idx = self.cache
        N, C, H, W = x_shape
        dx = xp.zeros(x_shape, dtype=dout.dtype)
        ni = xp.arange(N)[:, None]
        ci = xp.arange(C)[None, :]

        for i in range(self.oH):
            for j in range(self.oW):
                h_idx = max_h_idx[:, :, i, j]   # (N, C) — where was the max?
                w_idx = max_w_idx[:, :, i, j]   # (N, C)
                dx[ni, ci, h_idx, w_idx] += dout[:, :, i, j]

        return dx
    
class AdaptiveAvgPool2d(Layer):
    def __init__(self, out_size):
        super().__init__()
        self.oH, self.oW = (out_size, out_size) if isinstance(out_size, int) else out_size
        self.cache = None

    @staticmethod
    def _bins(in_sz, out_sz):
        return [(int(math.floor(i*in_sz/out_sz)),
                 int(math.ceil((i+1)*in_sz/out_sz))) for i in range(out_sz)]
    
    def forward(self, x):
        N, C, H, W = x.shape
        hb = self._bins(H, self.oH)
        wb = self._bins(W, self.oW)
        out = xp.zeros((N, C, self.oH, self.oW), dtype=x.dtype)
        for i, (h0, h1) in enumerate(hb):
            for j, (w0, w1) in enumerate(wb):
                out[:,:,i,j] = x[:,:,h0:h1,w0:w1].mean(axis=(2,3))
        self.cache = (x.shape, hb, wb)
        return out

    def backward(self, dout):
        x_shape, hb, wb = self.cache
        dx = xp.zeros(x_shape, dtype=dout.dtype)
        for i,(h0,h1) in enumerate(hb):
            for j,(w0,w1) in enumerate(wb):
                area = (h1-h0)*(w1-w0)
                dx[:,:,h0:h1,w0:w1] += dout[:,:,i:i+1,j:j+1] / area
        return dx


class Dropout(Layer):
    def __init__(self, p=0.5): super().__init__(); self.p = p; self.cache = None
    def forward(self, x):
        if not self.training or self.p == 0: return x
        mask = (xp.random.rand(*x.shape).astype(xp.float32) > self.p) / (1-self.p)
        self.cache = mask
        return x * mask
    def backward(self, d):
        return d if (not self.training or self.p == 0) else d * self.cache


class Flatten(Layer):
    def __init__(self): super().__init__(); self.cache = None
    def forward(self, x):  self.cache = x.shape; return x.reshape(x.shape[0], -1)
    def backward(self, d): return d.reshape(self.cache)


class Linear(Layer):
    def __init__(self, in_f, out_f):
        super().__init__()
        W = xp.random.randn(out_f, in_f).astype(xp.float32) * math.sqrt(2.0/in_f)
        self.params = {'W': W,                'b': xp.zeros(out_f, xp.float32)}
        self.grads  = {'W': xp.zeros_like(W), 'b': xp.zeros(out_f, xp.float32)}
        self.cache  = None

    def forward(self, x):
        self.cache = x
        return x @ self.params['W'].T + self.params['b']

    def backward(self, d):
        self.grads['W'] = d.T @ self.cache
        self.grads['b'] = d.sum(0)
        return d @ self.params['W']


# ══════════════════════════════════════════════════════════════════════════════
# Sequential container
# ══════════════════════════════════════════════════════════════════════════════

class Sequential:
    def __init__(self, *layers): self.layers = list(layers)
    def forward(self, x):
        for l in self.layers: x = l.forward(x)
        return x
    def backward(self, d):
        for l in reversed(self.layers): d = l.backward(d)
        return d
    def train_mode(self):
        for l in self.layers: l.train_mode()
    def eval_mode(self):
        for l in self.layers: l.eval_mode()
    def param_grad_pairs(self):
        out = []
        for l in self.layers: out.extend(l.param_grad_pairs())
        return out

# ══════════════════════════════════════════════════════════════════════════════
# CNNModel
# ══════════════════════════════════════════════════════════════════════════════

class CNNModel:
    def __init__(self, num_classes):
        self.features = Sequential(
            # ── Block 1 ──────────────────────────────────
            Conv2d(1, 32, 3, padding=1), BatchNorm2d(32), ReLU(),
            MaxPool2d(2,2),
            # ── Block 2 ──────────────────────────────────
            Conv2d(32, 64, 3, padding=1), BatchNorm2d(64), ReLU(),
            MaxPool2d(2, 2),
            # ── Block 3 ──────────────────────────────────
            Conv2d(64, 128, 3, padding=1), BatchNorm2d(128), ReLU(),
            AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = Sequential(
            Flatten(),
            Linear(128*4*4, 512), ReLU(), Dropout(0.40),
            Linear(512, 256),     ReLU(), Dropout(0.30),
            Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier.forward(self.features.forward(x))

    def backward(self, dout):
        self.features.backward(self.classifier.backward(dout))

    def train_mode(self):
        self.features.train_mode(); self.classifier.train_mode()

    def eval_mode(self):
        self.features.eval_mode(); self.classifier.eval_mode()

    def param_grad_pairs(self):
        return self.features.param_grad_pairs() + self.classifier.param_grad_pairs()

    def save(self, path):
        data = {}
        for i, layer in enumerate(self.features.layers + self.classifier.layers):
            for k, v in layer.params.items():
                data[f"l{i}_{k}"] = to_numpy(v)
            if isinstance(layer, BatchNorm2d):
                data[f"l{i}_rm"] = to_numpy(layer.running_mean)
                data[f"l{i}_rv"] = to_numpy(layer.running_var)
        np_cpu.savez(path, **data)
        print(f"  ✓ Best model saved → {path}.npz")

    def load(self, path):
        data = np_cpu.load(f"{path}.npz")
        for i, layer in enumerate(self.features.layers + self.classifier.layers):
            for k in layer.params:
                key = f"l{i}_{k}"
                if key in data: layer.params[k] = to_xp(data[key])
            if isinstance(layer, BatchNorm2d):
                if f"l{i}_rm" in data: layer.running_mean = to_xp(data[f"l{i}_rm"])
                if f"l{i}_rv" in data: layer.running_var  = to_xp(data[f"l{i}_rv"])

    def __repr__(self):
        def _block(name, seq):
            lines = [f"  ({name})"]
            for l in seq.layers:
                lines.append(f"    {l.__class__.__name__}")
            return "\n".join(lines)
        return "CNNModel(\n" + _block("features", self.features) \
             + "\n" + _block("classifier", self.classifier) + "\n)"

# ══════════════════════════════════════════════════════════════════════════════
# Loss  —  Cross-Entropy with optional label smoothing
# ══════════════════════════════════════════════════════════════════════════════

class CrossEntropyLoss:
    def __init__(self, label_smoothing=0.0):
        self.ls    = label_smoothing
        self.cache = None

    def __call__(self, logits, targets):
        N, C = logits.shape
        shifted = logits - logits.max(1, keepdims=True)
        probs   = xp.exp(shifted) / xp.exp(shifted).sum(1, keepdims=True)
        smooth  = xp.full((N, C), self.ls / C, dtype=xp.float32)
        smooth[xp.arange(N), targets] += 1.0 - self.ls
        loss = -(smooth * xp.log(probs + 1e-8)).sum(1).mean()
        self.cache = (probs, smooth, N)
        return float(to_numpy(loss))

    def backward(self):
        probs, smooth, N = self.cache
        return (probs - smooth) / N

# ══════════════════════════════════════════════════════════════════════════════
# Optimizer  —  AdamW
# ══════════════════════════════════════════════════════════════════════════════

class AdamW:
    def __init__(self, model, lr=0.001, betas=(0.9, 0.999), eps=1e-8, wd=1e-4):
        self.lr, self.b1, self.b2 = lr, betas[0], betas[1]
        self.eps, self.wd, self.t  = eps, wd, 0
        self.pg = model.param_grad_pairs()
        self.m  = [xp.zeros_like(p[n]) for p, g, n in self.pg]
        self.v  = [xp.zeros_like(p[n]) for p, g, n in self.pg]

    def step(self):
        self.t += 1
        bc1 = 1 - self.b1**self.t
        bc2 = 1 - self.b2**self.t
        for i, (p, g, n) in enumerate(self.pg):
            grad       = g[n]
            self.m[i]  = self.b1*self.m[i] + (1-self.b1)*grad
            self.v[i]  = self.b2*self.v[i] + (1-self.b2)*grad**2
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p[n] -= self.lr * (m_hat / (xp.sqrt(v_hat) + self.eps) + self.wd*p[n])

    def zero_grad(self):
        for p, g, n in self.pg: g[n][:] = 0

# ══════════════════════════════════════════════════════════════════════════════
# Scheduler  —  ReduceLROnPlateau
# ══════════════════════════════════════════════════════════════════════════════

class ReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', factor=0.5, patience=5):
        self.opt, self.mode  = optimizer, mode
        self.factor, self.patience = factor, patience
        self.best    = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0

    def step(self, metric):
        improved = (metric < self.best) if self.mode == 'min' else (metric > self.best)
        if improved:
            self.best = metric; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.opt.lr *= self.factor
                self.counter = 0
                print(f"  ↓ LR reduced to {self.opt.lr:.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# DataLoader
# ══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """Stores data on CPU; moves each batch to GPU (xp) on demand."""
    def __init__(self, x, y, batch_size=32, shuffle=True, s=None):
        self.x, self.y, self.s = x, y, s
        self.bs, self.shuffle = batch_size, shuffle
        self.n = x.shape[0]

    def __iter__(self):
        idx = np_cpu.random.permutation(self.n) if self.shuffle else np_cpu.arange(self.n)
        for start in range(0, self.n, self.bs):
            bi       = idx[start:start+self.bs]
            x_batch  = to_xp(self.x[bi])                          # CPU → GPU here
            y_batch  = (xp.asarray if GPU else np_cpu.asarray)(self.y[bi])
            src_batch = self.s[bi].tolist() if self.s is not None else None
            yield x_batch, y_batch, src_batch

    def __len__(self):
        return math.ceil(self.n / self.bs)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Data preparation
# ══════════════════════════════════════════════════════════════════════════════

def data_preparation(train_x, train_y, val_x, val_y, test_x, test_y,
                     batch_size=32,
                     train_s=None, val_s=None, test_s=None):
    print(f"Using device: {'GPU (CuPy)' if GPU else 'CPU (NumPy)'}")

    # ── Keep full dataset on CPU — batches are moved to GPU inside run_epoch ──
    def prep_x(a):
        return a.transpose(0, 3, 1, 2).astype(np_cpu.float32)   # (N,C,H,W) CPU

    def prep_y(a):
        return a.astype(np_cpu.int32)                            # CPU

    xs = [prep_x(a) for a in (train_x, val_x, test_x)]
    ys = [prep_y(a) for a in (train_y, val_y, test_y)]
    ss = [train_s, val_s, test_s]

    print(f"train_x shape: {xs[0].shape}")
    assert xs[0].shape[1] == 1, f"Expected 1 channel, got {xs[0].shape[1]}"

    loaders = (
        DataLoader(xs[0], ys[0], batch_size, shuffle=True,  s=ss[0]),
        DataLoader(xs[1], ys[1], batch_size, shuffle=False, s=ss[1]),
        DataLoader(xs[2], ys[2], batch_size, shuffle=False, s=ss[2]),
    )
    num_classes = len(np_cpu.unique(ys[0]))
    print(f"Number of classes: {num_classes}")
    return loaders, num_classes

# ══════════════════════════════════════════════════════════════════════════════
# 2. Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def _clear_caches(model):
    """Explicitly free all cached intermediate tensors from forward pass."""
    for layer in model.features.layers + model.classifier.layers:
        if hasattr(layer, 'cache'):
            layer.cache = None


def run_epoch(model, loader, criterion, optimizer=None):
    """One train or eval epoch. Pass optimizer=None for eval."""
    is_train = optimizer is not None
    model.train_mode() if is_train else model.eval_mode()

    total_loss, preds_all, targets_all = 0.0, [], []

    for inputs, targets, _ in loader:
        if is_train: optimizer.zero_grad()

        logits = model.forward(inputs)
        loss   = criterion(logits, targets)

        if is_train:
            dout = criterion.backward()
            model.backward(dout)
            optimizer.step()

        total_loss += loss
        preds_all.extend(to_numpy(xp.argmax(logits, axis=1)).tolist())
        targets_all.extend(to_numpy(targets).tolist())

        # ── Free ALL intermediate tensors after each batch ──────────────
        _clear_caches(model)
        del inputs, targets, logits
        if is_train: del dout
        if GPU: xp.get_default_memory_pool().free_all_blocks()

    return total_loss / len(loader), accuracy_score(targets_all, preds_all)


def check_early_stopping(val_loss, best_val_loss, counter, patience,
                         model, save_path):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save(save_path)
        counter = 0
    else:
        counter += 1
    return best_val_loss, counter, counter >= patience

# ══════════════════════════════════════════════════════════════════════════════
# 3. Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          epochs, patience, save_path="best_model") -> dict:
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    best_val_loss, counter = float("inf"), 0

    for epoch in range(epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion)

        scheduler.step(va_loss)

        for k, v in zip(["train_loss","val_loss","train_acc","val_acc"],
                         [tr_loss, va_loss, tr_acc, va_acc]):
            history[k].append(v)

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train — Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}")
        print(f"  Val   — Loss: {va_loss:.4f}  Acc: {va_acc:.4f}")

        best_val_loss, counter, stop = check_early_stopping(
            va_loss, best_val_loss, counter, patience, model, save_path)
        if stop:
            print("\nEarly stopping triggered!"); break

    return history

# ══════════════════════════════════════════════════════════════════════════════
# 4. Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, test_loader, criterion, save_path="best_model", scenario="model")-> list[list]:
    model.load(save_path)
    test_loss, test_acc = run_epoch(model, test_loader, criterion)

    preds_all, targets_all, sources_all, probs_all = [], [], [], []
    model.eval_mode()
    for inputs, targets, sources in test_loader:
        logits    = model.forward(inputs)
        logits_np = to_numpy(logits)
        shifted   = logits_np - logits_np.max(1, keepdims=True)
        exp       = np_cpu.exp(shifted)
        probs_np  = exp / exp.sum(1, keepdims=True)
        preds_all.extend(np_cpu.argmax(logits_np, axis=1).tolist())
        targets_all.extend(to_numpy(targets).tolist())
        sources_all.extend(sources if sources is not None else [])
        probs_all.extend(probs_np[:, 1].tolist())

        del inputs, targets, logits
        _clear_caches(model)
        if GPU: xp.get_default_memory_pool().free_all_blocks()

    precision = precision_score(targets_all, preds_all, average="weighted", zero_division=0)
    recall    = recall_score(targets_all,    preds_all, average="weighted", zero_division=0)
    f1        = f1_score(targets_all,        preds_all, average="weighted", zero_division=0)
    roc_auc   = roc_auc_score(targets_all,   probs_all)

    print("\n" + "="*40)
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print(f"ROC-AUC       : {roc_auc:.4f}")
    print("="*40)
    print("\nClassification Report:")
    print(classification_report(targets_all, preds_all,
                                target_names=["Human", "AI"], zero_division=0, digits=4))

    if sources_all:
        _print_source_breakdown(targets_all, preds_all, sources_all)

    plot_roc_curve(targets_all, probs_all, save_path=f"{scenario}_roc_curve.png")

    return preds_all, targets_all, sources_all


def _print_source_breakdown(targets, preds, sources) -> None:
    """Per-source error breakdown: how many AI/Human clips were mis-classified."""
    groups = defaultdict(lambda: {"total": 0, "wrong": 0,
                                               "wrong_as_human": 0, "wrong_as_ai": 0})
    for t, p, s in zip(targets, preds, sources):
        g = groups[s]
        g["total"] += 1
        if t != p:
            g["wrong"] += 1
            if p == 0: g["wrong_as_human"] += 1
            else:      g["wrong_as_ai"]    += 1

    print("\n" + "="*60)
    print("Per-Source Error Breakdown")
    print("="*60)
    print(f"{'Source':<25} {'Total':>6} {'Wrong':>6} {'Acc':>7} {'→Human':>8} {'→AI':>6}")
    print("-"*60)
    for src, g in sorted(groups.items()):
        acc = (g["total"] - g["wrong"]) / g["total"] if g["total"] else 0
        print(f"{src:<25} {g['total']:>6} {g['wrong']:>6} "
              f"{acc:>7.2%} {g['wrong_as_human']:>8} {g['wrong_as_ai']:>6}")
    print("="*60)

def _print_source_breakdown(targets, preds, sources):
    """Per-source error breakdown: how many AI/Human clips were mis-classified."""
    groups = defaultdict(lambda: {"total": 0, "wrong": 0,
                                               "wrong_as_human": 0, "wrong_as_ai": 0})
    for t, p, s in zip(targets, preds, sources):
        g = groups[s]
        g["total"] += 1
        if t != p:
            g["wrong"] += 1
            if p == 0: g["wrong_as_human"] += 1
            else:      g["wrong_as_ai"]    += 1

    print("\n" + "="*60)
    print("Per-Source Error Breakdown")
    print("="*60)
    print(f"{'Source':<25} {'Total':>6} {'Wrong':>6} {'Acc':>7} {'→Human':>8} {'→AI':>6}")
    print("-"*60)
    for src, g in sorted(groups.items()):
        acc = (g["total"] - g["wrong"]) / g["total"] if g["total"] else 0
        print(f"{src:<25} {g['total']:>6} {g['wrong']:>6} "
              f"{acc:>7.2%} {g['wrong_as_human']:>8} {g['wrong_as_ai']:>6}")
    print("="*60)


def _print_source_breakdown(targets, preds, sources):
    """Per-source error breakdown: how many AI/Human clips were mis-classified."""
    groups = defaultdict(lambda: {"total": 0, "wrong": 0,
                                               "wrong_as_human": 0, "wrong_as_ai": 0})
    for t, p, s in zip(targets, preds, sources):
        g = groups[s]
        g["total"] += 1
        if t != p:
            g["wrong"] += 1
            if p == 0: g["wrong_as_human"] += 1
            else:      g["wrong_as_ai"]    += 1

    print("\n" + "="*60)
    print("Per-Source Error Breakdown")
    print("="*60)
    print(f"{'Source':<25} {'Total':>6} {'Wrong':>6} {'Acc':>7} {'→Human':>8} {'→AI':>6}")
    print("-"*60)
    for src, g in sorted(groups.items()):
        acc = (g["total"] - g["wrong"]) / g["total"] if g["total"] else 0
        print(f"{src:<25} {g['total']:>6} {g['wrong']:>6} "
              f"{acc:>7.2%} {g['wrong_as_human']:>8} {g['wrong_as_ai']:>6}")
    print("="*60)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot_history(history, save_path="training_history.png"):
    epochs_ran = range(1, len(history["train_loss"]) + 1)
    fig, axes  = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (m0,m1), title, ylabel in zip(
        axes,
        [("train_loss","val_loss"), ("train_acc","val_acc")],
        ["Loss per Epoch", "Accuracy per Epoch"],
        ["Loss", "Accuracy"],
    ):
        ax.plot(epochs_ran, history[m0], label=f"Train {ylabel}", color="royalblue")
        ax.plot(epochs_ran, history[m1], label=f"Val {ylabel}",   color="tomato")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True)
    plt.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    folder = Path("images"); folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder/save_path, dpi=150); plt.show()


def plot_confusion_matrix(targets, preds, labels, save_path="confusion_matrix.png"):
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 14})
    plt.title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    folder = Path("images"); folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder/save_path, dpi=150)
    plt.show()


def plot_roc_curve(targets, probs, save_path="roc_curve.png"):
    """Plot ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(targets, probs)
    auc          = roc_auc_score(targets, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="royalblue", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0,1], [0,1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Human vs AI Music", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right"); plt.grid(True)
    plt.tight_layout()
    folder = Path("images")
    folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder/save_path, dpi=150)
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 6. Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def build_training_components(num_classes):
    model     = CNNModel(num_classes)
    criterion = CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model, lr=0.001, wd=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    print(model)
    return model, criterion, optimizer, scheduler


def run_pipeline(train_x, train_y, val_x, val_y, test_x, test_y,
                 labels, epochs=50, patience=10, batch_size=32, scenario="mel",
                 train_s=None, val_s=None, test_s=None):
    """End-to-end pipeline: prep → train → evaluate → visualise.
    Pass train_s, val_s, test_s (string arrays from get_feature)
    to enable the per-source error breakdown after evaluation.
    """
    np_cpu.random.seed(42)
    if GPU: xp.random.seed(42)

    (train_loader, val_loader, test_loader), num_classes = data_preparation(
        train_x, train_y, val_x, val_y, test_x, test_y, batch_size,
        train_s=train_s, val_s=val_s, test_s=test_s)

    model, criterion, optimizer, scheduler = build_training_components(num_classes)

    history = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=epochs, patience=patience, save_path=f"best_{scenario}_model")

    preds, targets, sources = evaluate(
        model, test_loader, criterion,
        save_path=f"best_{scenario}_model", scenario=scenario)

    plot_history(history,          save_path=f"{scenario}_training_history.png")
    plot_confusion_matrix(targets, preds, labels,
                          save_path=f"{scenario}_confusion_matrix.png")