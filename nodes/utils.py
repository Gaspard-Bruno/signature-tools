import torch
import numpy as np

def img_np_to_tensor(img_np_list):
    out_list = []
    for img_np in img_np_list:
        out_list.append(torch.from_numpy(img_np.astype(np.float32) / 255.0))
    return torch.stack(out_list)

def img_tensor_to_np(img_tensor):
    img_tensor = img_tensor.clone()
    img_tensor = img_tensor * 255.0
    mask_list = [x.squeeze().numpy().astype(np.uint8) for x in torch.split(img_tensor, 1)]
    return mask_list
