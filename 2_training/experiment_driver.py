'''
Driver script to train multiple models consecutively with different configs

work in progress and currently not operational
'''

from itertools import product

def generate_params(keys, *param_lists):
    combos = product(*param_lists)
    cfgs_list = []
    for combo in combos:
        cfgs_list.append(dict(zip(keys, combo)))

    return cfgs_list


def data_reduction_training_experiments(temp_model, base_dataset):
    # data reduction hyperparameters to test and find optimal configs of
    ratios = [0.1, 0.25, 0.5, 0.8, 1.0]
    warm_start_epochs = [1, 2, 3, 5, 10]
    selection_intervals = [1, 2, 3, 5, 10]
    subbatch_sizes = [100, 500, 1000]

    # assemble permutations of testable params
    experiment_configs = generate_params(
        ratios=ratios,
        warm_start_epochs=warm_start_epochs,
        selection_intervals=selection_intervals,
        subbatch_sizes=subbatch_sizes,
        keys=['subset_ratio', 'num_warm_start_epochs', 'subbatch_size', 'epoch_selection_interval']
    )

    # iterate over param combinations and train on each
    for exp_cfg in experiment_configs:
        # set data reduction config params to current test params
        dr_config.subset_ratio = exp_cfg['subset_ratio']
        dr_config.num_warm_start_epochs = exp_cfg['num_warm_start_epochs']
        dr_config.epoch_selection_interval = exp_cfg['epoch_selection_interval']
        dr_config.subbatch_size = exp_cfg['subbatch_size']
        
        # clear old stuff from memory for next iter
        gc.collect()
        torch.cuda.empty_cache()

        # set experiment info string for wandb run name and cache dir name
        experiment_info = f"ratio{dr_config.subset_ratio}_warm{dr_config.num_warm_start_epochs}-sel_int{dr_config.epoch_selection_interval}-subbatch{dr_config.subbatch_size}"
        
        # TODO: Implement logger, maybe W&B?
        # would also need to log/export the config data class for reference
        #logger.init(project_name=dr_config.wandb_project_name, run_name=experiment_info)

        temp_model = TreeModel()

        print(f"Running: {experiment_info}")
        train_data_reduction(
            temp_model,
            base_dataset,
            #logger,
        )
        #logger.finish()

        # delete objects to reinit next iter
        del base_dataset, dataset_info, temp_model_gp