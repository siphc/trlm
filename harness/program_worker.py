"""Runs inside the sandboxed subprocess. Imports a program and serves predict() over stdin/stdout.

Protocol (little-endian): request = int32 T, then T int32 tokens; T == -1 ends the session.
Response = int32 status (0 ok, 1 error), then for ok: int32 rows, int32 cols, rows*cols float32;
for error: int32 n, then n bytes of UTF-8 traceback.
"""
import importlib
import struct
import sys
import traceback


def main() -> None:
    program_parent, program_name = sys.argv[1], sys.argv[2]
    sys.path.insert(0, program_parent)
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    import numpy as np

    try:
        mod = importlib.import_module(program_name)
        predict = mod.predict
        err = None
    except Exception:
        err = traceback.format_exc()
    while True:
        head = inp.read(4)
        if len(head) < 4:
            return
        (T,) = struct.unpack("<i", head)
        if T == -1:
            return
        toks = np.frombuffer(inp.read(4 * T), dtype="<i4").astype(int).tolist()
        try:
            if err is not None:
                raise RuntimeError("program import failed:\n" + err)
            y = np.asarray(predict(toks), dtype=np.float64)
            if y.ndim != 2:
                raise ValueError(f"predict returned shape {y.shape}, expected 2-D")
            y32 = np.ascontiguousarray(y, dtype="<f4")
            out.write(struct.pack("<iii", 0, y32.shape[0], y32.shape[1]))
            out.write(y32.tobytes())
        except Exception:
            msg = traceback.format_exc().encode()
            out.write(struct.pack("<ii", 1, len(msg)))
            out.write(msg)
        out.flush()


if __name__ == "__main__":
    main()
