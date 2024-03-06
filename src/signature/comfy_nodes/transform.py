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
    def process(self,
                image: torch.Tensor,
                mask: torch.Tensor,
                padding: int):

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
    def process(self,
                image: torch.Tensor | None = None,
                mask: torch.Tensor | None = None,
                factor: float = 2.0,
                interpolation: str = 'nearest',
                antialias: bool = True):

        default_output = TensorImage(torch.zeros(1, 1, 1, 1)).get_comfy()
        output_image = default_output
        output_mask = default_output
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
    def process(self,
                image: torch.Tensor | None = None,
                mask: torch.Tensor | None = None,
                width:int = 512,
                height:int=512,
                keep_aspect_ratio: bool = False,
                interpolation: str = 'nearest',
                antialias: bool = True):

        default_output = TensorImage(torch.zeros(1, 1, 1, 1)).get_comfy()
        output_image = default_output
        output_mask = default_output

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
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
                "angle": ("FLOAT", {"default": 0.0, "min": 0, "max": 360.0, "step": 1.0}),
                "zoom_to_fit": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT

    def process(self, image: torch.Tensor | None = None, mask: torch.Tensor | None = None, angle: float = 0.0, zoom_to_fit: bool = False):
        default_output = TensorImage(torch.zeros(1, 1, 1, 1)).get_comfy()
        output_image = default_output
        output_mask = default_output

        if isinstance(image, torch.Tensor):
            img_tensor = TensorImage.from_comfy(image)
            angle_tensor = torch.tensor([angle], device=img_tensor.device, dtype=img_tensor.dtype)
            output_image = rotate(img_tensor, angle=angle_tensor)

            if zoom_to_fit:
                # Calculate new size to fit rotated image
                new_height = img_tensor.shape[-2] * torch.abs(torch.sin(angle_tensor)) + img_tensor.shape[-1] * torch.abs(torch.cos(angle_tensor))
                new_width = img_tensor.shape[-1] * torch.abs(torch.sin(angle_tensor)) + img_tensor.shape[-2] * torch.abs(torch.cos(angle_tensor))
                new_size = (int(new_height.max()), int(new_width.max()))

                # Resize the rotated image to fit
                output_image = resize(output_image, size=new_size)

            output_image = TensorImage(output_image).get_comfy()

        if isinstance(mask, torch.Tensor):
            mask_tensor = TensorImage.from_comfy(mask)
            angle_tensor = torch.tensor([angle], device=mask_tensor.device, dtype=mask_tensor.dtype)
            output_mask = rotate(mask_tensor, angle=angle_tensor)

            if zoom_to_fit:
                # Calculate new size to fit rotated mask
                new_height = mask_tensor.shape[-2] * torch.abs(torch.sin(angle_tensor)) + mask_tensor.shape[-1] * torch.abs(torch.cos(angle_tensor))
                new_width = mask_tensor.shape[-1] * torch.abs(torch.sin(angle_tensor)) + mask_tensor.shape[-2] * torch.abs(torch.cos(angle_tensor))
                new_size = (int(new_height.max()), int(new_width.max()))

                # Resize the rotated mask to fit
                output_mask = resize(output_mask, size=new_size)

            output_mask = TensorImage(output_mask).get_comfy()

        return (output_image, output_mask,)

class Cutout:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("rgb", "rgba")
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT

    def process(self, image: torch.Tensor, mask: torch.Tensor):
        tensor_image = TensorImage.from_comfy(image)
        tensor_mask = TensorImage.from_comfy(mask)

        if tensor_image.shape != tensor_mask.shape:
            tensor_image = resize(tensor_image, size=tensor_mask.shape[-2:])

        num_channels = tensor_image.shape[1]
        if num_channels == 4:
            tensor_image = rgba_to_rgb(tensor_image)
            num_channels = 3

        comfy_image_rgba = torch.cat((tensor_image, tensor_mask), dim=1)
        comfy_image_rgb = tensor_image.clone()
        comfy_image_rgb = comfy_image_rgb * tensor_mask.repeat(1, num_channels, 1, 1)
       

        comfy_image_rgb = TensorImage(comfy_image_rgb).get_comfy()
        comfy_image_rgba = TensorImage(comfy_image_rgba).get_comfy()

        return comfy_image_rgb, comfy_image_rgba

NODE_CLASS_MAPPINGS = {
    "Cutout": Cutout,
    "Rotate": Rotate,
    "Rescale": Rescale,
    "Resize": Resize,
    "Auto Crop": AutoCrop,
}