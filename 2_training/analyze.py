# entry point for various post-hoc analysis of training
# will expand later

import argparse
from pathlib import Path

from analysis.vram import vram_diag

if __name__ == '__main__':
    vram_log_fp = Path("/ofo-share/repos/brandon/tree-species-prediction/2_training/ckpts/0122-104529-gradmatch_r0.1_0122-104527/vram-log.jsonl")

    # PLACEHOLDER (currently vram_diag does nothing)
    vram_diag(vram_log_fp)