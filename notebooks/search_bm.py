#!/usr/bin/env python3
import os, pathlib, subprocess, time
from tqdm.auto import tqdm


NB = "1.8.3.3 nanogpt.ipynb"
OUTDIR = "executed"
PARAMS = [
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[2,2,8,8]", "SPARSE_SCHEDULE": "[1,1,2,2]"},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[2,2,6,6]", "SPARSE_SCHEDULE": "[1,1,2,2]"},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[2,2,8,8]", "SPARSE_SCHEDULE": "[1,1,4,4]"},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[2,2,6,6]", "SPARSE_SCHEDULE": "[1,1,4,4]"},

    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[2,2,2,8]", "SPARSE_SCHEDULE": "[1,1,2,2]"},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[2,8,8,8]", "SPARSE_SCHEDULE": "[1,1,2,2]"},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[8,8,8,8]", "SPARSE_SCHEDULE": "[1,1,2,2]"},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[2,2,2,8]", "SPARSE_SCHEDULE": "[2,2,2,2]"},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[2,8,8,8]", "SPARSE_SCHEDULE": "[2,2,2,2]"},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[8,8,8,8]", "SPARSE_SCHEDULE": "[2,2,2,2]"},

    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[2,4,6,8]", "SPARSE_SCHEDULE": "[1,1,2,2]"},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[2,4,6,8]", "SPARSE_SCHEDULE": "[2,2,2,2]"},

    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[4,4,4,8]", "SPARSE_SCHEDULE": "[1,1,2,2]"},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[4,4,8,8]", "SPARSE_SCHEDULE": "[1,1,2,2]"},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[4,8,8,8]", "SPARSE_SCHEDULE": "[1,1,2,2]"},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[4,4,4,8]", "SPARSE_SCHEDULE": "[2,2,2,2]"},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[4,4,8,8]", "SPARSE_SCHEDULE": "[2,2,2,2]"},
    {"SSEQ_LEN": 1024, "BATCH_SIZE": 4, "DENSE_SCHEDULE": "[4,8,8,8]", "SPARSE_SCHEDULE": "[2,2,2,2]"},
]

os.makedirs(OUTDIR, exist_ok=True)
base = pathlib.Path(NB).stem  # e.g., "1.8.3.2 nanogpt"

for i, p in tqdm(enumerate(PARAMS)):
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
        time.sleep(5)
    except Exception as e:
        print(f"[WARN] Run {i} failed: {e}; continuing...")
        time.sleep(10)