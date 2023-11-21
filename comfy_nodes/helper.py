import torch
import numpy as np

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