"""Calibration uses 1,024 training images; never the test set."""
import torch
@torch.inference_mode()
def calibrate(model,images,count=1024):
    model.eval()
    for batch in images[:count].split(128): model(batch)
    return model
