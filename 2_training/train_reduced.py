import gc
import time
from itertools import product

import torch
from models.TreeSpeciesClassifier import TreeSpeciesClassifierFromPretrained
from torch.utils.data import DataLoader, Subset
from training_utils.gather_gradients import gather_discriminator_gradients
from training_utils.io import save_model
from training_utils.omp import omp_select
from training_utils.train_epoch import train_epoch

from configs.data_reduction_config import dr_config
from configs.model_config import model_config


def generate_params(**params):
    """
    Yields dictionaries of parameter combinations dynamically
    Example usage:
    for combo in generate_params(reduction_ratio=[0.25, 0.5], warm_start_epoch=[100, 200]):
        ... do stuf with combo ...
    """
    keys, values = zip(*params.items()) if params else ([], [])
    for combo in product(*values):
        yield dict(zip(keys, combo))

def gradsel_reduce(
        temp_model,
        base_dataset,
        reduction_ratio,
        subbatch_size,
        device
    ):
    # gather gradients of Discriminator on real data (since that's what we are subsetting)
    grad_mat, grad_sum, subbatch_idxs = gather_discriminator_gradients(temp_model, base_dataset, device, subbatch_size, temp_model.batch_size)
    
    # determine budget based on reduction ratio and subbatch size'
    # this is how many subbatches worth of samples we can take
    budget = int(reduction_ratio * grad_mat.shape[0])
    print(budget, grad_mat.size())

    print("Solving OMP system... (choosing best gradient approximating subsets)")
    
    # run Orthogonal Matching Pursuit to determine best idxs using gradient matrix and gradient sum
    new_subbatch_idxs = omp_select(grad_mat, grad_sum, budget, device)
    print(f"Chose: {len(new_subbatch_idxs)} subbatches ({len(new_subbatch_idxs) * subbatch_size}) out of {len(base_dataset)}")

    # get actual sample idxs from subbatch idxs
    new_idxs = []
    for subbatch_i in new_subbatch_idxs:
        # extend with sample idxs from idxs of chosen subsets
        new_idxs.extend(subbatch_idxs[subbatch_i])

    # Rebuild the subset/dataloader with the new indices
    current_subset = Subset(base_dataset, new_idxs)
    print(f"Dataset size: {len(current_subset)}")

    del grad_mat, grad_sum, subbatch_idxs
    gc.collect()
    torch.cuda.empty_cache()

    return current_subset

def train_data_reduction(
        temp_model,
        base_dataset,
        full_data,
        logger,
    ):
    """
    Essentially should be same training loop as base model,
    while subsetting data ever `selection_interval` epochs,
    and some extra performance metrics for progress analysis
    """
    reduction_ratio = dr_config.subset_ratio
    selection_interval = dr_config.epoch_selection_interval
    num_warm_start_epochs = dr_config.num_warm_start_epochs
    subbatch_size = dr_config.subbatch_size

    # ensure subbatch budget won't be 0, 
    # i.e. |_(reduction ratio * num_batches)_| > 0
    assert int(reduction_ratio * len(base_dataset)) >= 1, f"ERROR: Reduction ratio too small for dataset size!"
    print(f"Will select subsets every {selection_interval} epochs, warm start: {num_warm_start_epochs}.")

    current_indices = list(range(len(base_dataset)))
    current_subset = Subset(base_dataset, current_indices)
    previous_subset = Subset(base_dataset, current_indices) # used to compare current and previous selections

    dataloader = DataLoader(
        current_subset,
        batch_size=model_config.batch_size,
        shuffle=True,
        pin_memory=True,
        #num_workers=4,
        #worker_init_fn=initializers.worker_init_fn
    )

    # loop over epochs
    for epoch_idx in range(temp_model.cur_epoch, model_config.epochs):
        # Train exactly one epoch
        train_epoch(temp_model, dataloader, epoch_idx)

        # TODO: compute a metric for training progress evaluation
        metric = -1
        
        # Subset selection every 'selection_interval' epochs
        # the selection starts at warm start, and recurs every selection interval AFTER warm start
        epoch_num = epoch_idx + 1
        if (epoch_num <= model_config.epochs and # if we haven't trained for the full spec'd num epochs and,
            epoch_num >= num_warm_start_epochs and # if we have trained enough on the full data and,
            (epoch_num - num_warm_start_epochs) % selection_interval == 0): # we are at a selection interval epoch, 

            # update subset selections
            print("*** Performing data reduction ***")   
            t0_subset = time.time()         
            current_subset = gradsel_reduce(
                temp_model,
                base_dataset,
                reduction_ratio,
                subbatch_size,
                dr_config.device
            )
            # TODO: impelement jaccard_sim to ensure subsets change each selection
            # (otherwise this only needs to be done once)
            # jaccard similarity compares how different set A (previously chosen subset) is from set B (newly chosen subset)
            
            #jaccard_sim = compute_pairwise_jaccard_similarity(previous_subset.indices, current_subset.indices)
            #print(f"New subset is {jaccard_sim*100:.4f}% similar to previous subset")
            #previous_subset = current_subset
            
            # TODO: implement metric logging, maybe W&B?
            # log metric
            '''logger.log({
                "jaccard_similarity": jaccard_sim, # similarity of cur and prev subsets
            }, step=epoch_idx)'''

            # clear old stuff from memory for next iter
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()

            dataloader = DataLoader( # update dataloader with new subset for next epoch
                current_subset,
                batch_size=int(reduction_ratio * temp_model.batch_size),
                shuffle=True,
                pin_memory=True,
                #num_workers=4,
                #worker_init_fn=initializers.worker_init_fn
            )

            subset_sel_time = time.time() - t0_subset # track time for selecting subset of data
        
        else:
            subset_sel_time = float('nan') # set selection time to nan if not selecting subsets

        # TODO: log metrics
        #logger.log({
        #    "metric": metric,
        #    "selection_interval_marker": subset_sel_time
        #}, step=epoch_idx)

        # TODO: save model every 10 epochs
        #metric = f"{metric:.4f}"
        #if epoch_num % 10 == 0 and epoch_num > 0:
        #    save_model(epoch_idx, metric)

def data_reduction_training_experiments(temp_model, base_dataset):
    # data reduction hyperparameters to test and find optimal configs of
    ratios = [0.05, 0.1, 0.25, 0.5, 0.8, 1.0]
    warm_start_epochs = [1000, 500, 300, 150, 50, 5]
    selection_intervals = [5, 10, 25, 50]
    subbatch_sizes = [100, 500, 1000]

    # assemble permutations of testable params
    experiment_configs = generate_params(
        ratios=ratios,
        warm_start_epochs=warm_start_epochs,
        selection_intervals=selection_intervals,
        subbatch_sizes=subbatch_sizes
    )

    # iterate over param combinations and train on each
    for exp_cfg in experiment_configs:
        # set data reduction config params to current test params
        dr_config.subset_ratio = exp_cfg['ratios']
        dr_config.num_warm_start_epochs = exp_cfg['warm_start_epochs']
        dr_config.epoch_selection_interval = exp_cfg['selection_intervals']
        dr_config.subbatch_size = exp_cfg['subbatch_sizes']
        
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