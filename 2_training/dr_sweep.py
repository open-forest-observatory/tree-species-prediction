import subprocess
import itertools
from datetime import datetime

# -----------------------
# Experiment grid
# -----------------------
SUBSET_RATIOS = [0.1, 0.25, 0.5]
METHODS = ["gradmatch", "random"]
EPOCHS = 10                   # default training epochs

# any extra args you want to globally enforce
COMMON_ARGS = {
    "--use_data_reduction": True,
    "--epochs": EPOCHS,
    "--max_class_imbalance_factor": 0,
    "--min_samples_per_class": 500,
    "--max_total_samples": 8000,
    "--use_class_balancing": False,
    "--data_split_level": EPOCHS,
}

# -----------------------
# Helper
# -----------------------
def run_experiment(method, ratio):
    tag = f"{method}_r{ratio}_{datetime.now().strftime('%m%d-%H%M%S')}"

    cmd = [
        "python", "train.py",
        "--strategy", method,
        "--subset_ratio", str(ratio),
        "--ckpt_dir_tag", tag,
    ]

    for k, v in COMMON_ARGS.items():
        cmd.append(k)
        if v is not None:
            cmd.append(str(v))

    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    for ratio, method, seed in itertools.product(SUBSET_RATIOS, METHODS):
        run_experiment(method, ratio, seed)
