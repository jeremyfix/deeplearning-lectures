# coding: utf-8

# Standard imports
import os

# External imports
import torch
import numpy as np
import matplotlib.pyplot as plt

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir:str, train: bool = True) -> None:
        super().__init__()

        if train:
            self.data = torch.from_numpy(np.load(os.path.join(root_dir, 'training_data.pkl'),
                                                    allow_pickle=True))
        else:
            self.data = torch.from_numpy(np.load(os.path.join(root_dir, 'testing_data.pkl'),
                                                    allow_pickle=True))
        # The data is 
        # :3  : ray_origins 
        # 3:6 : ray_directions
        # 6:  : gt_px_values
        self.dim_input = 6
        self.dim_output = 3

    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, index: int):
        sample = self.data[index]
        ray_origin = sample[:3]
        ray_direction = sample[3:6]
        gt_px_value = sample[6:]
        return (ray_origin, ray_direction), gt_px_value

def test_scene(root_dir):
    dataset = SceneDataset(root_dir)
    print(f"Dataset size: {len(dataset)}")
    idx = np.random.randint(0, len(dataset))
    (ray_origin, ray_direction), gt_px_value = dataset[idx]
    print(f"Sample {idx}:")
    print(f"  Ray origin: {ray_origin}")
    print(f"  Ray direction: {ray_direction}")
    print(f"  Ground truth pixel value: {gt_px_value}")

if __name__ == "__main__":
    root_dir = "/opt/datasets/NERF/Synth_Vandegar"
    test_scene(root_dir)