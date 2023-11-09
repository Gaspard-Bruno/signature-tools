import torch
import numpy as np

def image_array_to_torch(image_np):
    out_list = []
    img = image_np.get_float_value().transpose((2, 0, 1))
    out_list.append(torch.tensor(img, dtype=torch.float32))
    return torch.stack(out_list)


def img_np_to_tensor(img_np_list):
    out_list = []
    for img_np in img_np_list:
        out_list.append(torch.from_numpy(img_np.astype(np.float32) / 255.0))
    return torch.stack(out_list)