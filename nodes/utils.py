import torch
import numpy as np

def image_array_to_torch(image_np):
    out_list = []
    img = image_np.get_float_value().transpose((2, 0, 1))
    out_list.append(torch.tensor(img, dtype=torch.float32))
    return torch.stack(out_list)