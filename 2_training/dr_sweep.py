import subprocess
import itertools
from datetime import datetime

''' TODO:
Experiments:
- train with dr on various ratios with no fine tuning of backbone on large but not all data
    - give us an idea
- same as above but with all data

- train with dr at one ratio and full data to test n_classifier_layers
- same as above but with tunable backbone weights


'''

# Experiment grid
SUBSET_RATIOS = [0.1, 0.25, 0.5]
METHODS = ["gradmatch", "random"]
EPOCHS = 10
#MAX_SAMPLES = 5_000
MAX_SAMPLES = 0

# args not pertaining to data reduction
BASE_ARGS = {
    "--epochs": EPOCHS,
    "--max_class_imbalance_factor": 0,
    "--min_samples_per_class": 500,
    "--max_total_samples": MAX_SAMPLES,
    "--use_class_balancing": False,
    "--will_log_vram": True
}

def run_experiment(ratio, method=''):
    tag = f"drv4-allCroppedTreeLvlSplit-{method}_r{ratio}"

    cmd = [
        "python", "train.py",
        "--use_data_reduction" if method != '' else '',
        "--strategy", method,
        "--subset_ratio", str(ratio),
        "--ckpt_dir_tag", tag,
    ]

    for k, v in BASE_ARGS.items():
        cmd.append(k)
        if v is not None:
            cmd.append(str(v))

    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    for ratio, method in itertools.product(SUBSET_RATIOS, METHODS):
        run_experiment(ratio, method)
    run_experiment(1.0) # baseline all data
