import torch

def get_parameter_count(model, non_embedding=False):
    """
    Returns the total number of trainable parameters in the model.
    """
    if non_embedding:
        return model.get_num_params(non_embedding=True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_params(count):
    if count >= 1e6:
        return f"{count / 1e6:.2f}M"
    if count >= 1e3:
        return f"{count / 1e3:.2f}K"
    return str(count)
