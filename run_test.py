"""Execute a notebook source's code cells in order (one namespace), as Kaggle would.
Usage: python run_test.py src/01_dataset_acquisition.py   (env vars configure fixture paths)"""
import sys, time, traceback, matplotlib
matplotlib.use("Agg")
from build import code_cells

path = sys.argv[1]
ns = {"__name__": "__main__"}
for i, code in enumerate(code_cells(path), 1):
    t0 = time.time()
    try:
        exec(compile(code, f"{path}::cell{i}", "exec"), ns)
    except SystemExit:
        raise
    except Exception:
        print(f"\n!!! FAILED in cell {i}\n" + "-" * 60)
        print(code[:400])
        traceback.print_exc()
        sys.exit(1)
    print(f"--- cell {i} ok ({time.time()-t0:.1f}s)")
print(f"\nALL CELLS PASSED: {path}")
