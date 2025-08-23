#!/usr/bin/env python3
import os, pathlib, subprocess

NB = "1.8.3.2 nanogpt.ipynb"
OUTDIR = "executed"
PARAMS = [
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 16},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 8},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 2},

    {"SSEQ_LEN": 2048, "BATCH_SIZE": 8},
    {"SSEQ_LEN": 2048, "BATCH_SIZE": 4},
    {"SSEQ_LEN": 2048, "BATCH_SIZE": 2},

    {"SSEQ_LEN": 512, "BATCH_SIZE": 32},
    {"SSEQ_LEN": 512, "BATCH_SIZE": 16},
    {"SSEQ_LEN": 512, "BATCH_SIZE": 8},
    {"SSEQ_LEN": 512, "BATCH_SIZE": 4},
    {"SSEQ_LEN": 512, "BATCH_SIZE": 2},

    {"SSEQ_LEN": 256, "BATCH_SIZE": 64},
    {"SSEQ_LEN": 256, "BATCH_SIZE": 32},
    {"SSEQ_LEN": 256, "BATCH_SIZE": 16},
    {"SSEQ_LEN": 256, "BATCH_SIZE": 8},
    {"SSEQ_LEN": 256, "BATCH_SIZE": 4},

    {"SSEQ_LEN": 128, "BATCH_SIZE": 128},
    {"SSEQ_LEN": 128, "BATCH_SIZE": 64},
    {"SSEQ_LEN": 128, "BATCH_SIZE": 32},
    {"SSEQ_LEN": 128, "BATCH_SIZE": 16},
    {"SSEQ_LEN": 128, "BATCH_SIZE": 8},
]

os.makedirs(OUTDIR, exist_ok=True)
base = pathlib.Path(NB).stem  # e.g., "1.8.3.2 nanogpt"

for i, p in enumerate(PARAMS):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in p.items()})  # set SSEQ_LEN, BATCH_SIZE, LR_INIT, RUN_NAME
    out = f"{base}_{i}.executed.ipynb"
    cmd = [
        "/usr/bin/python", "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", NB,
        "--output", out, "--output-dir", OUTDIR,
        "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.kernel_name=python3",
    ]
    print(f"Running {NB} with {p} -> {OUTDIR}/{out}")
    try:
        subprocess.run(cmd, env=env, check=True)
    except Exception as e:
        print(f"[WARN] Run failed for {p['RUN_NAME']}: {e}; continuing...")