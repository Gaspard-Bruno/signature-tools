import torch
from .categories import TRANSFORM_CAT
from  ..src.signature.img.tensor_image import TensorImage
from kornia.geometry.transform import rescale, resize
from kornia.geometry.bbox import bbox_generator
from kornia.color import rgb_to_rgba, rgba_to_rgb
from torchvision.ops import masks_to_boxes
import numpy as np


def extract_bbox(mask: torch.Tensor):
    plain_mask = mask.squeeze(0)

    bbox = masks_to_boxes(plain_mask)
    h, w = plain_mask

    x = torch.any(mask, dim=1)
    y = torch.any(mask, dim=2)

    x_min = torch.argmax(x.float(), dim=1)
    x_max = h - torch.argmax(x.float().flip(dims=[1]), dim=1)

    y_min = torch.argmax(y.float(), dim=1)
    y_max = w - torch.argmax(y.float().flip(dims=[1]), dim=1)

    x_start = torch.tensor([x_min, x_max])
    y_start = torch.tensor([y_min, y_max])
    width = torch.tensor([0, w])
    height = torch.tensor([0, h])
    box = bbox_generator(x_start, y_start, width, height)

    return box


class AutoCropImage:

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "padding": ("INT", {"default": 0}),
            }}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "x", "y", "width", "height")

    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self, image: torch.Tensor, mask: torch.Tensor, padding: int):
        img_tensor = TensorImage.from_comfy(image)
        mask_tensor = TensorImage.from_comfy(mask)
        img_tensor = rgb_to_rgba(img_tensor, 1.0)

        B, C, H, W = img_tensor.shape
        mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=(H, W), mode='nearest')[:,0,:,:]
        MB, _, _ = mask_tensor.shape

        if MB < B:
            assert(B % MB == 0)
            mask_tensor = mask_tensor.repeat(B // MB, 1, 1)

        is_empty = ~torch.gt(torch.max(torch.reshape(mask_tensor,[MB, H * W]), dim=1).values, 0.)
        mask_tensor[is_empty,0,0] = 1.
        boxes = masks_to_boxes(mask_tensor)
        mask_tensor[is_empty,0,0] = 0.

        min_x = torch.clamp(boxes[:,0] - padding, min=0)
        min_y = torch.clamp(boxes[:,1] - padding, min=0)
        max_x = torch.clamp(boxes[:,2] + padding, max=W-1)
        max_y = torch.clamp(boxes[:,3] + padding, max=H-1)

        width = max_x - min_x + 1
        height = max_y - min_y + 1

        use_min_x = int(torch.min(min_x).item())
        use_min_y = int(torch.min(min_y).item())

        use_width = int(torch.max(width).item())
        use_height = int(torch.max(height).item())

        print(use_min_x, use_min_y, use_width, use_height)

        alpha_mask = torch.ones((B, C, H, W), device=mask_tensor.device)
        alpha_mask[:,3,:,:] = mask_tensor

        img_tensor = img_tensor * alpha_mask

        img_result = torch.zeros((B, 4, use_height, use_width), device=mask_tensor.device)
        mask_result = torch.zeros((B, 1, use_height, use_width), device=mask_tensor.device)
        for i in range(0, B):
            if not is_empty[i]:
                ymin = int(min_y[i].item())
                ymax = int(max_y[i].item())
                xmin = int(min_x[i].item())
                xmax = int(max_x[i].item())
                image_single = (img_tensor[i, :, ymin:ymax+1, xmin:xmax+1])
                mask_single = (mask_tensor[i, ymin:ymax+1, xmin:xmax+1])
                img_result[i] = image_single
                mask_result[i] = mask_single

        img_result = rgba_to_rgb(img_result)
        output_img = TensorImage(img_result).get_comfy()
        output_mask = TensorImage(mask_result).get_comfy()

        return (output_img, output_mask, use_min_x, use_min_y, use_width, use_height)


class RescaleImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "factor": ("FLOAT", {"default": 2.0, "min": 0.001, "max": 100.0, "step": 0.01}),
            "interpolation": (['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'],),
            "antialias": ("BOOLEAN", {"default": True},),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self, image: torch.Tensor, factor, interpolation, antialias):
        image = image.transpose(3, 1)
        output = rescale(image, factor=factor, interpolation=interpolation, antialias=antialias)
        output = output.transpose(3, 1)
        return (output,)

class ResizeImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "width": ("INT", {"default": 512}),
            "height": ("INT", {"default": 512}),
            "interpolation": (['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'],),
            "antialias": ("BOOLEAN", {"default": True},),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self, image: torch.Tensor, width, height, interpolation, antialias):
        image = image.transpose(3, 1)
        output = resize(image, size=(width, height), interpolation=interpolation, antialias=antialias)
        output = output.transpose(3, 1)
        return (output,)
class RescaleMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "mask": ("MASK",),
            "factor": ("FLOAT", {"default": 2.0, "min": 0.001, "max": 100.0, "step": 0.01}),
            "interpolation": (['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'],),
            "antialias": ("BOOLEAN", {"default": True},),
            }}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self, mask: torch.Tensor, factor, interpolation, antialias):
        #mask = mask.transpose(3, 1)
        mask = torch.stack([mask])
        output = rescale(mask, factor=factor, interpolation=interpolation, antialias=antialias)
        output = output[0]
        return (output,)

class ResizeMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "mask": ("MASK",),
            "width": ("INT", {"default": 512}),
            "height": ("INT", {"default": 512}),
            "interpolation": (['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'],),
            "antialias": ("BOOLEAN", {"default": True},),
            }}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self, mask: torch.Tensor, width, height, interpolation, antialias):
        mask = torch.stack([mask])
        output = resize(mask, size=(width, height), interpolation=interpolation, antialias=antialias)
        output = output[0]
        return (output,)

NODE_CLASS_MAPPINGS = {
    "Rescale Image": RescaleImage,
    "Resize Image": ResizeImage,
    "Rescale Mask": RescaleMask,
    "Resize Mask": ResizeMask,
    "Auto Crop Image": AutoCropImage,
}