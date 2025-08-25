#!/usr/bin/env python3
import os, pathlib, subprocess, time
from tqdm.auto import tqdm


NB = "1.8.3.3 nanogpt.ipynb"
OUTDIR = "executed"
PARAMS = [
    {"SCHEDULE": [
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
            {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
    ]},
    {"SCHEDULE": [
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 16,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 16,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 16,
            'lr_mult': 1.0,
        },
    ]},
    {"SCHEDULE": [
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 4,
            'lr_mult': 1.0,
        },
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 4,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
    ]},
    {"SCHEDULE": [
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 4,
            'lr_mult': 1.0,
        },
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 4,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 4,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 4,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 4,
            'lr_mult': 1.0,
        },
    ]},
    {"SCHEDULE": [
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 4,
            'lr_mult': 1.0,
        },
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 16,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 16,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 16,
            'lr_mult': 1.0,
        },
    ]},
    {"SCHEDULE": [
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 2,
            'lr_mult': 1.0,
        },
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 4,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 6,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
    ]},
    {"SCHEDULE": [
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 2,
            'lr_mult': 1.0,
        },
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 4,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 4,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
    ]},
    {"SCHEDULE": [
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 4,
            'lr_mult': 1.0,
        },
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 4,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 16,
            'lr_mult': 1.0,
        },
    ]},
    {"SCHEDULE": [
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 2, 'sparse': 1,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 8,
            'lr_mult': 1.0,
        },
        {
            'dense': 8, 'sparse': 2,
            'seq_len': 1024 * 6, 'batch_size': 16,
            'lr_mult': 1.0,
        },
    ]},
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