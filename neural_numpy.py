from __future__ import annotations

from typing import List

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class MLP:
    def __init__(self, sizes: List[int], rng: np.random.Generator, output_scale: float = 1.0):
        self.weights = []
        self.biases = []
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            if i == len(sizes) - 2:
                scale *= output_scale
            self.weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float64))
        self.zero_grad()

    def zero_grad(self) -> None:
        self.grad_w = [np.zeros_like(w) for w in self.weights]
        self.grad_b = [np.zeros_like(b) for b in self.biases]

    def forward(self, x: np.ndarray, cache: bool = False):
        a = x
        activations = [a]
        preacts = []
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = a @ w + b
            preacts.append(z)
            a = np.tanh(z)
            activations.append(a)
        out = a @ self.weights[-1] + self.biases[-1]
        if cache:
            return out, (activations, preacts)
        return out

    def backward(self, cache, grad_out: np.ndarray, accumulate: bool = True) -> np.ndarray:
        activations, preacts = cache
        grad = grad_out
        last = len(self.weights) - 1
        if accumulate:
            self.grad_w[last] += activations[-1].T @ grad
            self.grad_b[last] += grad.sum(axis=0, keepdims=True)
        grad = grad @ self.weights[last].T
        for i in range(last - 1, -1, -1):
            grad = grad * (1.0 - np.tanh(preacts[i]) ** 2)
            if accumulate:
                self.grad_w[i] += activations[i].T @ grad
                self.grad_b[i] += grad.sum(axis=0, keepdims=True)
            grad = grad @ self.weights[i].T
        return grad

    def copy_from(self, other: "MLP") -> None:
        for dst, src in zip(self.weights, other.weights):
            dst[...] = src
        for dst, src in zip(self.biases, other.biases):
            dst[...] = src

    def soft_update_from(self, other: "MLP", tau: float) -> None:
        for dst, src in zip(self.weights, other.weights):
            dst[...] = (1.0 - tau) * dst + tau * src
        for dst, src in zip(self.biases, other.biases):
            dst[...] = (1.0 - tau) * dst + tau * src

    def params_and_grads(self):
        params = []
        grads = []
        for w, g in zip(self.weights, self.grad_w):
            params.append(w)
            grads.append(g)
        for b, g in zip(self.biases, self.grad_b):
            params.append(b)
            grads.append(g)
        return params, grads


class Adam:
    def __init__(self, net: MLP, lr: float, grad_clip: float = 10.0):
        self.net = net
        self.lr = lr
        self.grad_clip = grad_clip
        params, _ = net.params_and_grads()
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self) -> None:
        params, grads = self.net.params_and_grads()
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
        scale = 1.0
        if norm > self.grad_clip:
            scale = self.grad_clip / (norm + 1e-12)
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            g = g * scale
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * (g * g)
            m_hat = self.m[i] / (1.0 - 0.9**self.t)
            v_hat = self.v[i] / (1.0 - 0.999**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def grad_norm(net: MLP) -> float:
    _, grads = net.params_and_grads()
    return float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))


class Actor:
    def __init__(
        self,
        input_dim: int,
        hidden: List[int],
        rng: np.random.Generator,
        initial_action: float = 0.75,
        fixed_input_skip: np.ndarray | None = None,
        trainable_skip_basis: np.ndarray | None = None,
    ):
        self.net = MLP([input_dim, *hidden, 1], rng, output_scale=0.10)
        self.fixed_input_skip = (
            np.zeros((input_dim, 1), dtype=np.float64)
            if fixed_input_skip is None
            else np.asarray(fixed_input_skip, dtype=np.float64).reshape(input_dim, 1)
        )
        if trainable_skip_basis is None:
            self.trainable_skip_basis = np.zeros((input_dim, 0), dtype=np.float64)
        else:
            basis = np.asarray(trainable_skip_basis, dtype=np.float64)
            if basis.ndim != 2 or basis.shape[0] != input_dim:
                raise ValueError("trainable_skip_basis must have shape (input_dim, profile_dim)")
            self.trainable_skip_basis = basis.copy()
        self.profile_coefficients = np.zeros((self.trainable_skip_basis.shape[1], 1), dtype=np.float64)
        self.profile_grad = np.zeros_like(self.profile_coefficients)
        initial_action = float(np.clip(initial_action, 1e-3, 1.0 - 1e-3))
        self.net.biases[-1][0, 0] = np.log(initial_action / (1.0 - initial_action))

    def zero_grad(self) -> None:
        self.net.zero_grad()
        self._ensure_profile_state()
        self.profile_grad.fill(0.0)

    def _ensure_profile_state(self) -> None:
        # Older pickled actors predate the optional trainable profile.
        if not hasattr(self, "trainable_skip_basis"):
            self.trainable_skip_basis = np.zeros((self.fixed_input_skip.shape[0], 0), dtype=np.float64)
        if not hasattr(self, "profile_coefficients"):
            self.profile_coefficients = np.zeros((self.trainable_skip_basis.shape[1], 1), dtype=np.float64)
        if not hasattr(self, "profile_grad"):
            self.profile_grad = np.zeros_like(self.profile_coefficients)

    def forward(self, features: np.ndarray, cache: bool = False):
        self._ensure_profile_state()
        raw, net_cache = self.net.forward(features, cache=True)
        raw = raw + features @ self.fixed_input_skip
        if self.profile_coefficients.size:
            raw = raw + features @ self.trainable_skip_basis @ self.profile_coefficients
        sig = sigmoid(raw)
        if cache:
            return sig, (net_cache, sig, np.asarray(features, dtype=np.float64))
        return sig

    def backward(self, cache, grad_action: np.ndarray) -> None:
        net_cache, sig, features = cache
        grad_raw = grad_action * sig * (1.0 - sig)
        self.net.backward(net_cache, grad_raw, accumulate=True)
        if self.profile_coefficients.size:
            self.profile_grad += self.trainable_skip_basis.T @ features.T @ grad_raw

    def copy_profile_from(self, other: "Actor") -> None:
        self._ensure_profile_state()
        other._ensure_profile_state()
        if self.profile_coefficients.shape != other.profile_coefficients.shape:
            raise ValueError("actor profile shapes do not match")
        self.profile_coefficients[...] = other.profile_coefficients

    def soft_update_profile_from(self, other: "Actor", tau: float) -> None:
        self._ensure_profile_state()
        other._ensure_profile_state()
        if self.profile_coefficients.shape != other.profile_coefficients.shape:
            raise ValueError("actor profile shapes do not match")
        self.profile_coefficients[...] = (
            (1.0 - tau) * self.profile_coefficients + tau * other.profile_coefficients
        )
