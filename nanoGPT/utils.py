import torch
def save_checkpoint(model, path):
    try:
        torch.save(model.state_dict(), path)
    except Exception as e:
        raise(f"Exception occured when saving checkpoint {e}")

def load_checkpoint(model_skeleton, path):
    try:
        state_dict = torch.load(path, weights_only=True)
        model_skeleton.load_state_dict(state_dict)
    except Exception as e:
        raise(f"Exception occured when loading model checkpoint: {e}")
    return model_skeleton