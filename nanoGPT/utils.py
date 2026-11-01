import torch
def save_checkpoint(model, path):
    try:
        torch.save(model.state_dict, path)
    except Exception as e:
        raise(f"Exception occured when saving checkpoint {e}")
    