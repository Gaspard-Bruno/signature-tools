import torch
import numpy as np

def numpy_to_tensor(img_np):
    return torch.from_numpy(img_np.astype(np.float32) / 255.0)

def tensor_to_numpy(img_tensor):
    img_tensor = img_tensor.clone()
    img_tensor = img_tensor * 255.0
    return img_tensor.squeeze().numpy().astype(np.uint8)
