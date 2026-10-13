"""Shared harness utilities. READ-ONLY for replacement agents.

- ProgramRunner: runs a program's predict() in a subprocess. On macOS it uses sandbox-exec to make
  models/ and data/ unreadable and to block network access. Elsewhere it says it cannot enforce this.
- Model access: fp32 log-probabilities of a trained model (the fidelity target), optional head ablation.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB = 2000
CTX = 256


# ----------------------------------------------------------------------------- program runner

class ProgramError(RuntimeError):
    pass


class ProgramRunner:
    def __init__(self, program_dir: str, sandbox: bool = True):
        program_dir = os.path.abspath(program_dir)
        self.program_dir = program_dir
        parent, name = os.path.dirname(program_dir), os.path.basename(program_dir)
        worker = os.path.join(REPO, "harness", "program_worker.py")
        cmd = [sys.executable, "-B", "-I", worker, parent, name]
        self.sandbox_status = "off (requested)"
        self._profile = None
        if sandbox:
            if sys.platform == "darwin" and shutil.which("sandbox-exec"):
                prof = (
                    "(version 1)\n(allow default)\n"
                    f'(deny file-read* (subpath "{os.path.join(REPO, "models")}"))\n'
                    f'(deny file-read* (subpath "{os.path.join(REPO, "data")}"))\n'
                    f'(deny file-read* (subpath "{os.path.join(REPO, "logs")}"))\n'
                    f'(deny file-read* (subpath "{os.path.join(REPO, "analysis")}"))\n'
                    "(deny network*)\n"
                )
                fd, self._profile = tempfile.mkstemp(suffix=".sb")
                with os.fdopen(fd, "w") as f:
                    f.write(prof)
                cmd = ["sandbox-exec", "-f", self._profile] + cmd
                self.sandbox_status = "enforced (sandbox-exec: models/, data/, logs/, analysis/ unreadable; no network)"
            else:
                self.sandbox_status = "NOT ENFORCED: sandbox-exec unavailable on this OS"
                print("WARNING: harness cannot make models/ unreadable on this OS.", file=sys.stderr)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "1", "HOME": "/nonexistent"}
        if not sandbox:  # self-test programs may need to know which model to wrap
            env.update({k: v for k, v in os.environ.items() if k.startswith("TRLM_")})
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
                                     cwd=tempfile.gettempdir())

    def predict(self, tokens) -> np.ndarray:
        toks = np.asarray(tokens, dtype="<i4")
        self.proc.stdin.write(struct.pack("<i", len(toks)) + toks.tobytes())
        self.proc.stdin.flush()
        head = self._read(4)
        (status,) = struct.unpack("<i", head)
        if status != 0:
            (n,) = struct.unpack("<i", self._read(4))
            raise ProgramError(self._read(n).decode(errors="replace"))
        r, c = struct.unpack("<ii", self._read(8))
        return np.frombuffer(self._read(4 * r * c), dtype="<f4").reshape(r, c).astype(np.float64)

    def _read(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise ProgramError(f"program process died (exit {self.proc.poll()})")
            buf += chunk
        return buf

    def close(self) -> None:
        try:
            self.proc.stdin.write(struct.pack("<i", -1))
            self.proc.stdin.flush()
        except Exception:
            pass
        self.proc.wait(timeout=30)
        if self._profile:
            os.remove(self._profile)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def program_sha(program_dir: str) -> str:
    h = hashlib.sha256()
    for root, dirs, files in sorted(os.walk(program_dir)):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for fn in sorted(files):
            if fn.endswith(".py"):
                p = os.path.join(root, fn)
                h.update(os.path.relpath(p, program_dir).encode())
                h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def validate_rows(y: np.ndarray, T: int) -> None:
    if y.shape != (T, VOCAB):
        raise ProgramError(f"predict returned shape {y.shape}, expected ({T}, {VOCAB})")
    if np.isnan(y).any() or np.isposinf(y).any():
        raise ProgramError("predict returned NaN or +inf")
    m = y.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(y - m).sum(axis=1, keepdims=True)))[:, 0]
    bad = np.abs(lse) > 1e-3
    if bad.any():
        i = int(np.argmax(bad))
        raise ProgramError(f"row {i} is not a log-distribution: logsumexp = {lse[i]:.6f}")


def causality_check(runner: ProgramRunner, windows: np.ndarray, rng: np.random.Generator) -> None:
    """Perturb every token after position t; rows 0..t must not change."""
    for w in windows:
        x = w[:CTX].astype(int)
        base = runner.predict(x.tolist())
        for t in (0, 17, 128, CTX - 2):
            x2 = x.copy()
            x2[t + 1:] = rng.integers(0, VOCAB, size=CTX - t - 1)
            y2 = runner.predict(x2.tolist())
            if not np.allclose(base[: t + 1], y2[: t + 1], atol=1e-5, rtol=0):
                raise ProgramError(f"causality violated: changing tokens after {t} changed rows <= {t}")
        short = runner.predict(x[:5].tolist())
        validate_rows(short, 5)
        if not np.allclose(base[:5], short, atol=1e-5, rtol=0):
            raise ProgramError("causality violated: prefix of length 5 gives different rows")


# ----------------------------------------------------------------------------- model access

def model_path(model_id: str) -> str:
    if os.path.exists(model_id):
        return model_id
    return os.path.join(REPO, "models", "final", model_id, "final.pt")


def load_model(model_id: str):
    sys.path.insert(0, REPO)
    from src.train.train import load_model as _load

    return _load(model_path(model_id), "cpu")


def model_logprobs(model, windows: np.ndarray, bs: int = 25, ablate_heads=(), ablation="zero", means=None):
    """fp32 log-probs of shape (n, T, VOCAB) for windows (n, >=T). Inputs are windows[:, :CTX].

    ablate_heads: iterable of head indices (block 0) whose output z_h (before the output projection) is
    replaced:
      ablation="zero":    z_h = 0
      ablation="mean":    z_h = means["z"][h], its mean over reference windows (see head_means)
      ablation="earlier": z_h(i) = a_ii v_i + means["ctx"][h]: keep the head's current-token part, replace the
                          part coming from earlier tokens by its mean (see head_means)
    Mean and earlier-token ablation were added 2026-10-12 with human approval (logs/decisions.md).
    """
    import torch

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if ablate_heads and ablation == "earlier":
        return _logprobs_earlier_ablated(model, windows, list(ablate_heads), means["ctx"], bs)
    out = []
    handle = None
    if ablate_heads:
        H, Dh = model.cfg.n_heads, model.cfg.d_head
        heads = list(ablate_heads)
        if ablation == "mean":
            fill = torch.as_tensor(np.asarray(means["z"]), dtype=torch.float32)
        elif ablation != "zero":
            raise ValueError(ablation)

        def pre_hook(mod, args):
            z = args[0].clone()
            B, T, D = z.shape
            z = z.view(B, T, H, Dh)
            for h in heads:
                z[:, :, h, :] = 0.0 if ablation == "zero" else fill[h]
            return (z.view(B, T, D),)

        handle = model.blocks[0].attn.out.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            for i in range(0, len(windows), bs):
                x = torch.as_tensor(np.asarray(windows[i : i + bs])[:, :CTX].astype(np.int64))
                out.append(torch.log_softmax(model(x).float(), dim=-1).numpy())
    finally:
        if handle is not None:
            handle.remove()
    return np.concatenate(out).astype(np.float64)


def _attention_parts(model, x):
    """Explicit block-0 attention for a batch x (B, T): returns (x0, z, selfpart) with z = sum_j a_ij v_j and
    selfpart = a_ii v_i, both (B, T, H, Dh). Mirrors src/train/model.py exactly (pre-LN, no biases)."""
    import math

    import torch

    blk = model.blocks[0]
    B, T = x.shape
    H, Dh, D = model.cfg.n_heads, model.cfg.d_head, model.cfg.d_model
    x0 = model.wte(x) + model.wpe(torch.arange(T))[None]
    q, k, v = blk.attn.qkv(blk.ln1(x0)).split(D, dim=-1)
    q = q.view(B, T, H, Dh).transpose(1, 2)
    k = k.view(B, T, H, Dh).transpose(1, 2)
    v = v.view(B, T, H, Dh).transpose(1, 2)
    sc = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
    sc = sc.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    a = sc.softmax(-1)                                                     # (B, H, T, T)
    z = (a @ v).transpose(1, 2)                                            # (B, T, H, Dh)
    selfpart = (torch.diagonal(a, dim1=-2, dim2=-1)[..., None] * v).transpose(1, 2)
    return x0, z, selfpart


def _finish(model, x0, z):
    blk = model.blocks[0]
    B, T = z.shape[:2]
    h = x0 + blk.attn.out(z.reshape(B, T, -1))
    if blk.mlp is not None:
        h = h + blk.mlp(blk.ln2(h))
    return model.ln_f(h) @ model.wte.weight.T


def head_means(model, windows: np.ndarray, bs: int = 25) -> dict:
    """Means over all positions of `windows` of each head's output z_h and of its earlier-token part
    z_h - a_ii v_i; used by mean and earlier-token ablation. Returns {"z": (H, Dh), "ctx": (H, Dh)}."""
    import torch

    acc_z = acc_c = None
    n = 0
    with torch.no_grad():
        for i in range(0, len(windows), bs):
            x = torch.as_tensor(np.asarray(windows[i : i + bs])[:, :CTX].astype(np.int64))
            _, z, sp = _attention_parts(model, x)
            sz, sc = z.double().sum((0, 1)), (z - sp).double().sum((0, 1))
            acc_z = sz if acc_z is None else acc_z + sz
            acc_c = sc if acc_c is None else acc_c + sc
            n += x.numel()
    return {"z": (acc_z / n).float().numpy(), "ctx": (acc_c / n).float().numpy()}


def _logprobs_earlier_ablated(model, windows, heads, mean_ctx, bs):
    import torch

    out = []
    mc = torch.as_tensor(np.asarray(mean_ctx), dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(windows), bs):
            x = torch.as_tensor(np.asarray(windows[i : i + bs])[:, :CTX].astype(np.int64))
            x0, z, sp = _attention_parts(model, x)
            z = z.clone()
            for h in heads:
                z[:, :, h, :] = sp[:, :, h, :] + mc[h]
            out.append(torch.log_softmax(_finish(model, x0, z).float(), -1).numpy())
    return np.concatenate(out).astype(np.float64)


# ----------------------------------------------------------------------------- eval data

def eval_windows(split: str, model_id: str | None = None) -> np.ndarray:
    d = os.path.join(REPO, "data", "eval")
    if split == "dev_fast":
        return np.load(os.path.join(d, "dev_fast.npy"))
    if split == "dev_full":
        return np.load(os.path.join(d, "dev_full.npy"))
    if split == "test":
        return np.load(os.path.join(d, "test_full.npy"))
    if split == "claims":
        return np.load(os.path.join(d, "claims.npy"))
    if split == "probe":
        assert model_id is not None
        return np.load(os.path.join(d, "probe", f"{os.path.basename(model_id.rstrip('/'))}.npy"))
    raise ValueError(split)


def bigram_logprobs() -> np.ndarray:
    return np.load(os.path.join(REPO, "data", "baselines", "bigram_logprobs.npy")).astype(np.float64)


def git_state() -> tuple[str, bool]:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                           text=True).stdout.strip() or "none"
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                                    text=True).stdout.strip())
    except Exception:
        h, dirty = "none", True
    return h, dirty
