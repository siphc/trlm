"""Harness self-tests (phase 1). Run: python -m pytest -q harness/test_harness.py"""
import os
import textwrap

import numpy as np
import pytest

from harness.common import ProgramError, ProgramRunner, causality_check
from harness.lint_program import lint_program

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_program(tmp_path, body: str, name="prog"):
    d = tmp_path / name
    d.mkdir()
    (d / "__init__.py").write_text(textwrap.dedent(body))
    return str(d)


GOOD = """
import numpy as np

def predict(tokens):
    return np.full((len(tokens), 2000), -np.log(2000.0))
"""


def test_lint_accepts_clean_program(tmp_path):
    assert lint_program(_write_program(tmp_path, GOOD))["ok"]


@pytest.mark.parametrize("bad", [
    "import os\n",
    "import pickle\n",
    "from pathlib import Path\n",
    "x = open('f')\n",
    "x = eval('1')\n",
    "import numpy as np\nx = np.load('f')\n",
    "x = [" + ", ".join(str(i) for i in range(65)) + "]\n",
    "x = [[" + "], [".join(", ".join(str(i) for i in range(13)) for _ in range(5)) + "]]\n",
    "x = '" + "QUJD" * 30 + "'\n",
    "x = b'" + "a" * 65 + "'\n",
    "import numpy as np\nf = getattr(np, 'lo' + 'ad')\n",
    "x = ().__class__\n",
])
def test_lint_rejects(tmp_path, bad):
    r = lint_program(_write_program(tmp_path, GOOD + bad))
    assert not r["ok"], bad


def test_lint_allows_64_element_literal(tmp_path):
    body = GOOD + "x = [" + ", ".join(str(i) for i in range(64)) + "]\n"
    r = lint_program(_write_program(tmp_path, body))
    assert r["ok"] and r["largest_literal"] == 64


def test_sandbox_blocks_models_and_data(tmp_path):
    os.makedirs(os.path.join(REPO, "models"), exist_ok=True)
    body = f"""
    import numpy as np
    def predict(tokens):
        with open({os.path.join(REPO, 'data', 'prep_meta.json')!r}) as f:
            f.read()
        return np.full((len(tokens), 2000), -np.log(2000.0))
    """
    with ProgramRunner(_write_program(tmp_path, body), sandbox=True) as r:
        assert r.sandbox_status.startswith("enforced")
        with pytest.raises(ProgramError, match="Operation not permitted"):
            r.predict([1, 2, 3])
    body2 = body.replace("'data', 'prep_meta.json'", "'models'")
    body2 = f"""
    import numpy as np
    import os
    def predict(tokens):
        os.listdir({os.path.join(REPO, 'models')!r})
        return np.full((len(tokens), 2000), -np.log(2000.0))
    """
    with ProgramRunner(_write_program(tmp_path, body2, "prog2"), sandbox=True) as r:
        with pytest.raises(ProgramError, match="Operation not permitted"):
            r.predict([1, 2, 3])


def test_causality_check_catches_peeking(tmp_path):
    body = """
    import numpy as np
    def predict(tokens):
        out = np.full((len(tokens), 2000), -np.log(2000.0))
        out[:, tokens[-1]] += 1.0
        out -= np.log(np.exp(out).sum(1, keepdims=True))
        return out
    """
    wins = np.load(os.path.join(REPO, "data", "eval", "dev_fast.npy"))[:1]
    with ProgramRunner(_write_program(tmp_path, body)) as r:
        with pytest.raises(ProgramError, match="causality"):
            causality_check(r, wins, np.random.default_rng(0))


def test_good_program_passes_causality(tmp_path):
    wins = np.load(os.path.join(REPO, "data", "eval", "dev_fast.npy"))[:1]
    with ProgramRunner(_write_program(tmp_path, GOOD)) as r:
        causality_check(r, wins, np.random.default_rng(0))
