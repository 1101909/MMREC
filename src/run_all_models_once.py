# coding: utf-8
"""Run every configured model once on one dataset.

This is a smoke/compatibility runner: it collapses model hyper-parameter grids to
their first value and runs one epoch per model, writing a concise pass/fail log.
"""

import argparse
import datetime
import os
import subprocess
import sys
import traceback
import types

import yaml


MODELS = [
    "BM3",
    "BPR",
    "DAMRS",
    "DRAGON",
    "DualGNN",
    "FREEDOM",
    "GRCN",
    "ItemKNNCBF",
    "LATTICE",
    "LayerGCN",
    "LGMRec",
    "LightGCN",
    "MGCN",
    "MMGCN",
    "MVGAE",
    "PGL",
    "SELFCFED_LGN",
    "SLMRec",
    "SMORE",
    "VBPR",
]


def first_values_for_model(model):
    config_path = os.path.join(os.getcwd(), "configs", "model", "{}.yaml".format(model))
    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}

    hyper_parameters = set(config.get("hyper_parameters") or [])
    overrides = {}
    for key, value in config.items():
        if key == "hyper_parameters":
            continue
        if key in hyper_parameters and isinstance(value, list):
            overrides[key] = value[0] if value else None
    return overrides


def run_one(model, dataset, data_path=None):
    # The local Anaconda env has torch_geometric installed, but its ssl module is
    # broken. torch_geometric only imports ssl for downloader helpers here.
    if "ssl" not in sys.modules:
        sys.modules["ssl"] = types.ModuleType("ssl")

    from utils.quick_start import quick_start

    config_dict = {
        "gpu_id": 0,
        "use_gpu": False,
        "epochs": 1,
        "eval_step": 1,
        "stopping_step": 1,
        "hyper_parameters": ["seed"],
    }
    if data_path:
        config_dict["data_path"] = data_path
    config_dict.update(first_values_for_model(model))
    quick_start(model=model, dataset=dataset, config_dict=config_dict, save_model=False)


def run_all(dataset, models, timeout, data_path=None, tag=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = os.path.abspath(os.path.join(os.getcwd(), "run_all_results"))
    os.makedirs(result_dir, exist_ok=True)
    run_name = "{}-{}".format(dataset, tag) if tag else dataset
    summary_path = os.path.join(result_dir, "summary-{}-{}.tsv".format(run_name, timestamp))

    with open(summary_path, "w", encoding="utf-8") as summary:
        summary.write("model\tstatus\texit_code\tlog\n")
        for model in models:
            log_path = os.path.join(result_dir, "{}-{}-{}.log".format(model, run_name, timestamp))
            cmd = [sys.executable, __file__, "--model", model, "--dataset", dataset]
            if data_path:
                cmd.extend(["--data-path", data_path])
            with open(log_path, "w", encoding="utf-8") as log:
                log.write("Running {} on {}\n".format(model, dataset))
                log.flush()
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=os.getcwd(),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=timeout,
                    )
                    returncode = proc.returncode
                    status = "PASS" if returncode == 0 else "FAIL"
                except subprocess.TimeoutExpired:
                    returncode = 124
                    status = "TIMEOUT"
                    log.write("\nTimed out after {} seconds.\n".format(timeout))

            summary.write("{}\t{}\t{}\t{}\n".format(model, status, returncode, log_path))
            summary.flush()
            print("{}\t{}\t{}".format(model, status, log_path), flush=True)

    print("Summary: {}".format(summary_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--model")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--data-path")
    parser.add_argument("--tag")
    args = parser.parse_args()

    if args.model:
        try:
            run_one(args.model, args.dataset, args.data_path)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    else:
        run_all(args.dataset, args.models, args.timeout, args.data_path, args.tag)


if __name__ == "__main__":
    main()
