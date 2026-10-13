"""Create the black-box track's separate repository (AGENTS.md section 10).

  python scripts/setup_blackbox.py [--dest /Users/svi/Desktop/trlm_blackbox]

The black-box repo gets: AGENTS.md, configs/, the data (symlinks to this repo's token streams, evaluation
sets, tokenizer and train-split n-gram tables), the harness (patched so the model is reached only through
the logits server, no ablation), src/train/data.py (data loading only), the n-gram baselines,
scripts/iterate.py, a program/ skeleton with only the tokenizer vocabulary, and BLACKBOX.md. No models/, no
analysis/ tools, no open-track program, logs or notes.

The model is served by `python -m blackbox.server --model attn_s0` started from THIS (open) repo.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess

OPEN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLIENT = '''"""Black-box track: the only door to the model. Logits only.

    from blackbox.query_api import query_logits
    logits = query_logits([[0, 12, 345], [0, 7]])   # list of float32 arrays, shape (len_i, 2000)

Row i of each result is the model's next-token logits after tokens[: i + 1]. Sequences may be any length
1..256 and may be synthetic. The server (started by the orchestrator) answers for one model; every query
is logged on its side. Connection: TRLM_BB_ADDRESS (default /tmp/trlm_blackbox.sock), key TRLM_BB_KEY.
"""
from __future__ import annotations

import os
from multiprocessing.connection import Client

import numpy as np

_CONN = None
CHUNK = 256


def _conn():
    global _CONN
    if _CONN is None:
        _CONN = Client(os.environ.get("TRLM_BB_ADDRESS", "/tmp/trlm_blackbox.sock"), family="AF_UNIX",
                       authkey=os.environ.get("TRLM_BB_KEY", "trlm-blackbox").encode())
    return _CONN


def query_logits(batch_of_token_lists, model_id=None) -> list:
    seqs = [[int(t) for t in s] for s in batch_of_token_lists]
    out = []
    for i in range(0, len(seqs), CHUNK):
        _conn().send({"tokens": seqs[i : i + CHUNK]})
        resp = _conn().recv()
        if "error" in resp:
            raise ValueError(resp["error"])
        out.extend(np.asarray(x, dtype=np.float32) for x in resp["logits"])
    return out
'''

REMOTE_MODEL = '''

# ----------------------------------------------------------------------------- black-box model access
# This repository is the black-box track: the model is reached only through blackbox/query_api.py
# (logits only). load_model / model_logprobs below replace the open-track versions above.

class RemoteModel:
    def __init__(self, model_id: str):
        self.model_id = model_id


def load_model(model_id: str):  # noqa: F811
    return RemoteModel(model_id)


def model_logprobs(model, windows, bs: int = 64, ablate_heads=(), ablation="zero", means=None):  # noqa: F811
    if ablate_heads:
        raise RuntimeError("black-box track: logits only, no head ablation")
    from blackbox.query_api import query_logits

    out = []
    w = np.asarray(windows)[:, :CTX].astype(int)
    for i in range(0, len(w), bs):
        for lg in query_logits(w[i : i + bs].tolist()):
            lg = lg.astype(np.float64)
            lg = lg - lg.max(1, keepdims=True)
            out.append(lg - np.log(np.exp(lg).sum(1, keepdims=True)))
    return np.stack(out)


def head_means(*a, **k):  # noqa: F811
    raise RuntimeError("black-box track: logits only, no head ablation")
'''

GUIDE = '''# Black-box track

You are the black-box track of this project (AGENTS.md section 10). Your only access to the model is
`blackbox/query_api.py` (`query_logits(list of token lists)` -> logits). You may query any input, including
synthetic text. You have the data (`data/`), the harness (`harness/`, read-only) and AGENTS.md. There are no
model weights here and no analysis tools; build your own under `analysis/`.

- Target model: `attn_s0` (the server answers for it). Its architecture is described in AGENTS.md section 2.
- Write the replacement program in `program/` (interface: AGENTS.md section 7). `program/vocab.py` holds the
  tokenizer's 2000 token strings (tokenizer definition, not model information).
- Score with `python -m harness.eval_fidelity --model attn_s0 --split dev_fast` (KL through the server).
  Claim tests may use `harness.claims` (model log-probs only; ablation raises an error here).
- Log every harness run and commit (`scripts/iterate.py`), keep `logs/dead_ends.md` and `logs/surprises.md`.
- Use `.venv/bin/python`. The machine has one GPU (MPS) shared with other jobs and tight RAM and disk:
  keep checkpoints small and delete large intermediates.
- Do not read anything outside this repository.
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(os.path.dirname(OPEN), "trlm_blackbox"))
    args = ap.parse_args()
    D = args.dest
    assert not os.path.exists(D), f"{D} exists"
    os.makedirs(D)
    for f in ("AGENTS.md",):
        shutil.copy(os.path.join(OPEN, f), D)
    shutil.copytree(os.path.join(OPEN, "configs"), os.path.join(D, "configs"))
    shutil.copytree(os.path.join(OPEN, "harness"), os.path.join(D, "harness"),
                    ignore=shutil.ignore_patterns("__pycache__", "selftest"))
    shutil.copytree(os.path.join(OPEN, "baselines"), os.path.join(D, "baselines"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(D, "src", "train"))
    open(os.path.join(D, "src", "__init__.py"), "w").close()
    open(os.path.join(D, "src", "train", "__init__.py"), "w").close()
    shutil.copy(os.path.join(OPEN, "src", "train", "data.py"), os.path.join(D, "src", "train", "data.py"))
    os.makedirs(os.path.join(D, "scripts"))
    shutil.copy(os.path.join(OPEN, "scripts", "iterate.py"), os.path.join(D, "scripts", "iterate.py"))
    os.makedirs(os.path.join(D, "blackbox"))
    open(os.path.join(D, "blackbox", "__init__.py"), "w").close()
    with open(os.path.join(D, "blackbox", "query_api.py"), "w") as f:
        f.write(CLIENT)
    # harness: remote model access; sandbox also blocks the open repo's private folders
    common = os.path.join(D, "harness", "common.py")
    s = open(common).read()
    s = s.replace('''                    "(deny network*)\\n"''', '''                    + "".join(f'(deny file-read* (subpath "{os.path.join(%r, sub)}"))\\n'
                              for sub in ("models", "analysis", "program", "logs", "data")) +
                    "(deny network*)\\n"''' % OPEN)
    s += REMOTE_MODEL
    open(common, "w").write(s)
    # data: symlinks to the open repo's token streams, eval sets, tokenizer and train-split n-gram tables
    os.makedirs(os.path.join(D, "data"))
    for item in ("tokens", "eval", "baselines", "tokenizer.json", "prep_meta.json"):
        os.symlink(os.path.join(OPEN, "data", item), os.path.join(D, "data", item))
    os.symlink(os.path.join(OPEN, ".venv"), os.path.join(D, ".venv"))
    # program skeleton: vocabulary only, uniform predictor
    os.makedirs(os.path.join(D, "program", "tests"))
    shutil.copy(os.path.join(OPEN, "program", "vocab.py"), os.path.join(D, "program", "vocab.py"))
    with open(os.path.join(D, "program", "__init__.py"), "w") as f:
        f.write('"""Black-box replacement program (empty start: uniform distribution)."""\n'
                "import numpy as np\n\n\ndef predict(tokens):\n"
                "    return np.full((len(tokens), 2000), -np.log(2000.0))\n")
    os.makedirs(os.path.join(D, "logs"))
    for name, title in (("dead_ends.md", "Dead-end ledger"), ("surprises.md", "Surprises")):
        with open(os.path.join(D, "logs", name), "w") as f:
            f.write(f"# {title} (black-box track)\n")
    with open(os.path.join(D, "BLACKBOX.md"), "w") as f:
        f.write(GUIDE)
    with open(os.path.join(D, ".gitignore"), "w") as f:
        f.write("data\n.venv\n__pycache__/\n*.pyc\nanalysis/out/\nanalysis/scratch/\n")
    subprocess.run(["git", "init", "-q"], cwd=D, check=True)
    subprocess.run(["git", "add", "-A"], cwd=D, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "Black-box track: initial copy (data, harness via logits server, AGENTS.md)"],
                   cwd=D, check=True)
    print(f"created {D}")


if __name__ == "__main__":
    main()
