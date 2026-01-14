import subprocess

N_REPEAT_TRIALS = 3
BASE_TRAIN_CMD = ["python", "2_training/train.py"]
ARGS_TESTING =  {
    "--ckpt_dir_tag": "dr-v2t1-r",
    "--subset_ratio": 1.0,
    "--use_data_reduction": False,
}

#ratios = [0.1, 1.0, 'none']
ratios = [0.75, 0.5, 0.25, 0.1, 1.0, 'none']

if __name__ == '__main__':
    run_cmds = []
    for ratio in ratios:
        for i in range(N_REPEAT_TRIALS):
            cur_cmd = BASE_TRAIN_CMD.copy()
            cur_args = ARGS_TESTING.copy()
            cur_args['--ckpt_dir_tag'] = f"{ARGS_TESTING['--ckpt_dir_tag']}-noES-{str(ratio).upper()}-iter{i}"
            cur_args['--subset_ratio'] = 1.0 if str(ratio).lower() == 'none' else str(ratio)
            cur_args['--use_data_reduction'] = False if str(ratio).lower() == 'none' else True

            for k, v in cur_args.items():
                cur_cmd.extend([k, v])
            
            run_cmds.append(cur_cmd)

    print(f"Running {len(run_cmds)} model trainings...")
    print(run_cmds) # test

    for cmd in run_cmds:
        cmd = [str(elem) for elem in cmd]
        print(f"*** RUNNING: {' '.join(cmd)} ***")
        subprocess.run(cmd)

    
    