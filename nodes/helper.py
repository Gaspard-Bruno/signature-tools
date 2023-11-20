import torch

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