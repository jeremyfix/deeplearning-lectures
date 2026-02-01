# coding: utf-8

# Standard imports
import logging
import sys

# External imports
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import tqdm

# Local imports
import nirlab.utils as utils

def generate_sample(model, 
                    batch_size=32,
                    height=100, 
                    width=100):
    """
    Generate an image by sampling the model on a regular grid.
    
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    prev_training = model.training
    model.eval()
    
    dim_input = 2
    with torch.no_grad():
        X0 = torch.tensor([0]*dim_input, dtype=torch.float32, device=device).unsqueeze(0)
        out0 = model(X0)
        dim_output = out0.shape[-1]

        img = np.zeros((width*height, dim_output), np.float32)
       
        # Build a regular grid of pixels positions to sample
        points = utils.build_coordinate_Nd(width, height, device=device) # torch tensor (width*height, dim_input)
        
        # Build a dataloader from it
        batch_size = batch_size
        dataloader = torch.utils.data.DataLoader(points, batch_size=batch_size, shuffle=False, drop_last=False)

        # Loop over batches
        idx = 0
        for batch_points in tqdm.tqdm(dataloader):
            out_batch = model(batch_points)  # (B, dim_output)
            img[idx:idx+batch_points.shape[0], :] = out_batch.detach().cpu().numpy()
            idx += batch_points.shape[0]

        img = img.reshape((width, height, dim_output)).transpose((1, 0, 2))  # (height, width, dim_output)

        # Convert to [0, 255] range
        img = img * 255.0
        
        # Clip to [0, 255], and cast to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)

    if prev_training:
        model.train()
    return img

def sample_image(model, 
                 logdir, 
                 epoch, 
                 train_loader,
                 batch_size=40960,
                 height=1000, 
                 width=1000):
    filename = logdir / f"sample_epoch_{epoch}.png"
    sample = generate_sample(
        model,
        batch_size=batch_size,
        height=height,
        width=width,
    )
    save_arr = sample
    if save_arr.ndim == 3 and save_arr.shape[2] == 1:
        save_arr = save_arr[:, :, 0]
    Image.fromarray(save_arr).save(filename)
    logging.info(f"Saved sample image to {filename}")


def test_sampler():
    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            src_width = 4
            src_height = 4
            self.scale = torch.tensor([src_width, src_height])
            self.colors = torch.rand((src_width, src_height, 3))

        def forward(self, X):
            # X is (B, 2) in [0, 1]. 
            # Scale by width, height, and stick to ints
            X = (X*self.scale.unsqueeze(0)).floor().long()
            X = torch.clamp(X, torch.tensor(0), self.scale-1)
            out = self.colors[X[:, 0], X[:, 1], :]

            return out
            

    model = Model()
    sample_image(model, "img.png", height=80, width=80)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    test_sampler()
