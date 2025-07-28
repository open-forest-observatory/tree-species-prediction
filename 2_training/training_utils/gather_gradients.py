import torch
from tqdm import tqdm
import gc
import math
from models.model import TreeModel

# Function to chunk a list into pieces of size chunk_size
def chunk_list(lst, chunk_size):
    for start in range(0, len(lst), chunk_size):
        yield lst[start : start + chunk_size]

def gather_discriminator_gradients(
    temp_model, 
    dataset, 
    device,
    subbatch_size,
    batch_size=90000
):
    """
    3 deep for loops looks bad I know, but I swear it's still just O(n),
    there's just a lot of chunks since we have a batch size for model training,
    which is different than the subbatch size we use for a better gradient approximation.
    
    The alternatives are find gradients PER SAMPLE (essentially subbatch_size of 1) which would take longer than a full model training,
    Or choose best full batches to approximate the gradient, which is hugely lacking precision.
    And given our batch size it would mean there's only 10 choices, which means we couldn't subset < 10% of the original dataset

    Gathers per-sample gradient of the model.
    
    Returns:
        grads_matrix: (N, param_dim)
        grad_sum: (param_dim,)
    """
    # model in eval mode 
    temp_model.eval()

    # copy model so gradient gathering doesn't effect model's state dict
    model_copy = TreeModel(
        
    ).to(device)
    
    model_copy.load_state_dict(temp_model.state_dict())
    for p in model_copy.parameters():
        p.requires_grad = (True)

    subbatch_grads = [] # store gradients calculated for each subbatch
    subbatch_idxs = [] # store sample idxs of samples in subbatches relative to whole dset

    # idxs of all samples in dataset
    # idx tracking of samples is needed so we can subset samples in dataset correctly,
    # when using subbatch for gradient approximations
    all_idxs = list(range(len(dataset)))
    N = len(all_idxs)

    # iterate over idxs of batches
    # batch_idxs is a list of sample idxs of len batch_size
    for batch_idxs in tqdm(chunk_list(all_idxs, batch_size), total=math.ceil(N / batch_size), desc="Gathering gradients for batches"):
        
        # Now split batch_idxs into smaller slices of size minibatch_size
        for cur_subbatch_idxs in chunk_list(batch_idxs, subbatch_size):
            
            # get all samples of this current subbatch
            subbatch_samples_list = []
            for i in cur_subbatch_idxs:
                subbatch_samples_list.append(dataset[i])
            subbatch_samples = torch.stack(subbatch_samples_list, dim=0).to(device)

            # Zero old grads
            model_copy.zero_grad(set_to_none=True)
            
            # shape: (subbatch_size, 1); or  (subbatch_size % num_subbatches, 1) if last subbatch
            d_out = model_copy(subbatch_samples) # Forward pass for this sub-batch

            # WGAN real term = -mean(D(real)); not using mone as done in wgan train loop
            d_loss = -torch.mean(d_out)

            # Backprop
            d_loss.backward()

            # Flatten param grads
            grads_i = []
            for p in model_copy.parameters():
                if p.grad is not None:
                    grads_i.append(p.grad.view(-1))
            grads_i = torch.cat(grads_i)  # shape: (param_dim,)

            # get subbatch of grads of GPU to prevent OOM at a greater level of granularity
            grads_i = grads_i.detach().cpu()

            subbatch_grads.append(grads_i)
            subbatch_idxs.append(cur_subbatch_idxs)

    # list -> tensor
    grads_matrix = torch.stack(subbatch_grads, dim=0)  # shape: (N, param_dim)
    # sum grads across all samples
    grad_sum = grads_matrix.sum(dim=0)              # shape: (param_dim,)

    # delete model copy
    del model_copy
    gc.collect()
    torch.cuda.empty_cache()
    return grads_matrix, grad_sum, subbatch_idxs