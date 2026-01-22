# quick chatgpt script (I checked it for correctness) to analyze the vram logs produced when model_config.will_log_vram == True
# simply to help find the source of the oom issues I'm having

import pandas as pd

def vram_diag(log_path, top=20, min_count=1):
    pass