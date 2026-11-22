"""Convert percent-format sources (src/NN_*.py) into Kaggle .ipynb notebooks.
Cell markers:  '# %% [markdown]'  '# %%'  '# %% [common]' (inserts src/common_setup.py)  '# %% [lib]' (inserts src/exp_lib.py)."""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC, OUT = ROOT / "src", ROOT / "notebooks"
OUT.mkdir(exist_ok=True)
COMMON = (SRC / "common_setup.py").read_text().rstrip("\n")
LIB = (SRC / "exp_lib.py").read_text().rstrip("\n")

def parse(text):
    cells, kind, buf = [], None, []
    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l)
                             for l in body.split("\n"))
        if kind == "common":
            body = COMMON
        if kind == "lib":
            body = LIB
        if body.strip():
            cells.append((kind, body))
    for line in text.split("\n"):
        m = re.match(r"^# %%(?: \[(markdown|common|lib)\])?\s*$", line)
        if m:
            flush()
            kind, buf = (m.group(1) or "code"), []
        else:
            buf.append(line)
    flush()
    return cells

def to_ipynb(cells):
    nb_cells = []
    for kind, body in cells:
        src = [l + "\n" for l in body.split("\n")]
        src[-1] = src[-1].rstrip("\n")
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": src})
    return {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}

def code_cells(path):
    return [b for k, b in parse(Path(path).read_text()) if k in ("code", "common", "lib")]

if __name__ == "__main__":
    targets = sys.argv[1:] or sorted(p.name for p in SRC.glob("[0-9][0-9]*_*.py"))
    for name in targets:
        p = SRC / name
        nb = to_ipynb(parse(p.read_text()))
        out = OUT / (p.stem + ".ipynb")
        out.write_text(json.dumps(nb, indent=1))
        print(f"built {out.name}: {len(nb['cells'])} cells")
