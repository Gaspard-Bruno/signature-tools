import torch
from .categories import TRANSFORM_CAT
from ..img.tensor_image import TensorImage
from kornia.geometry.transform import rescale, resize, rotate
from kornia.color import rgb_to_rgba, rgba_to_rgb
from torchvision.ops import masks_to_boxes


class AutoCrop:

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


class Rescale:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                },
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
                "factor": ("FLOAT", {"default": 2.0, "min": 0.001, "max": 100.0, "step": 0.01}),
                "interpolation": (['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'],),
                "antialias": ("BOOLEAN", {"default": True}),
                },
            }
    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self, image: torch.Tensor | None = None, mask: torch.Tensor | None = None, factor: float = 2.0, interpolation: str = 'nearest', antialias: bool = True):
        output_image = torch.ones(1, 1, 1, 1)
        output_mask = torch.ones(1, 1, 1, 1)
        tuple_factor = (factor, factor)
        if isinstance(image, torch.Tensor):
            img_tensor = TensorImage.from_comfy(image)
            output_image = rescale(img_tensor, factor=tuple_factor, interpolation=interpolation, antialias=antialias)
            output_image = TensorImage(output_image).get_comfy()

        if isinstance(mask, torch.Tensor):
            mask_tensor = TensorImage.from_comfy(mask)
            output_mask = rescale(mask_tensor, factor=tuple_factor, interpolation=interpolation, antialias=antialias)
            output_mask = TensorImage(output_mask).get_comfy()

        return (output_image, output_mask,)


class Resize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "keep_aspect_ratio": ("BOOLEAN", {"default": False}),
                "interpolation": (['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'],),
                "antialias": ("BOOLEAN", {"default": True},),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self, image: torch.Tensor | None = None, mask: torch.Tensor | None = None, width:int = 512, height:int=512, keep_aspect_ratio: bool = False, interpolation: str = 'nearest', antialias: bool = True):
        output_image = torch.ones(1, 1, 1, 1)
        output_mask = torch.ones(1, 1, 1, 1)

        size = (height, width)
        if isinstance(image, torch.Tensor):
            img_tensor = TensorImage.from_comfy(image)

            if keep_aspect_ratio:
                image_height, image_width = img_tensor.shape[-2:]
                aspect_ratio = image_width / image_height

                if width > height:
                    new_width = width
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = height
                    new_width = int(new_height * aspect_ratio)

                size = (new_height, new_width)

            output_image = resize(img_tensor, size=size, interpolation=interpolation, antialias=antialias)
            output_image = TensorImage(output_image).get_comfy()

        if isinstance(mask, torch.Tensor):
            mask_tensor = TensorImage.from_comfy(mask)

            if keep_aspect_ratio:
                mask_height, mask_width = mask_tensor.shape[-2:]
                aspect_ratio = mask_width / mask_height

                if width > height:
                    new_width = width
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = height
                    new_width = int(new_height * aspect_ratio)

                size = (new_height, new_width)

            output_mask = resize(mask_tensor, size=size, interpolation=interpolation, antialias=antialias)
            output_mask = TensorImage(output_mask).get_comfy()

        return (output_image, output_mask,)

class Rotate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
                "angle": ("FLOAT", {"default": 0.0, "min": 0, "max": 360.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT

    def process(self, image: torch.Tensor | None = None, mask: torch.Tensor | None = None, angle: float = 0.0):

        output_image = torch.ones(1, 1, 1, 1)
        output_mask = torch.ones(1, 1, 1, 1)

        if isinstance(image, torch.Tensor):
            img_tensor = TensorImage.from_comfy(image)
            N = image.shape[0]
            angle_tensor = torch.tensor([angle]*N, device=img_tensor.device, dtype=img_tensor.dtype)
            output_image = rotate(img_tensor, angle=angle_tensor)
            output_image = TensorImage(output_image).get_comfy()

        if isinstance(mask, torch.Tensor):
            mask_tensor = TensorImage.from_comfy(mask)
            print(mask_tensor.shape)
            N = mask.shape[0]
            angle_tensor = torch.tensor([angle]*N, device=mask_tensor.device, dtype=mask_tensor.dtype)
            output_mask = rotate(mask_tensor, angle=angle_tensor)
            output_mask = TensorImage(output_mask).get_comfy()

        return (output_image, output_mask,)

NODE_CLASS_MAPPINGS = {
    "Rotate": Rotate,
    "Rescale": Rescale,
    "Resize": Resize,
    "Auto Crop": AutoCrop,
}