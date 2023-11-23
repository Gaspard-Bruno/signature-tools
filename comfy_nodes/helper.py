import torch
import numpy as np
import kornia as K

def comfy_img_to_torch(image: torch.Tensor) -> torch.Tensor:
    #ex [1, 2675, 3438, 3] -> [1, 3, 2675, 3438]
    return image.permute(0, 3, 1, 2)

def comfy_mask_to_torch(mask: torch.Tensor) -> torch.Tensor:
    #ex [1, 2675, 3438] -> [1, 1, 2675, 3438]
    if mask.ndim == 2:
        mask = torch.stack([mask])
    if mask.ndim == 3:
        mask = torch.stack([mask])
    return mask

def torch_img_to_comfy(image: torch.Tensor) -> torch.Tensor:
    #ex [1, 3, 2675, 3438] -> [1, 2675, 3438, 3]
    return image.permute(0, 2, 3, 1).cpu()

def torch_mask_to_comfy(mask: torch.Tensor) -> torch.Tensor:
    #ex [1, 1, 2675, 3438] -> [1, 2675, 3438]
    return mask[0].cpu()

def np_to_tensor(numpy_array) -> torch.Tensor:
    return torch.from_numpy(numpy_array.astype(np.float32) / 255.0)
    out_list = []
    for img_np in np_list:
        out_list.append()
    return torch.stack(out_list)

def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    tensor = torch_img_to_comfy(tensor)
    tensor = tensor.clone()
    np_tensor = (tensor.detach().cpu() * 255.0).numpy().astype(np.uint8)
    return np_tensor
    # np_list = [x.squeeze().numpy().astype(np.uint8) for x in torch.split(tensor, 1)]
    # return np_list