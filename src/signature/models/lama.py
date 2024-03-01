import torch
import torch.nn.functional as F
import kornia.geometry.transform as K
import kornia.morphology as M
from kornia.color import rgba_to_rgb
from kornia.utils import get_cuda_or_mps_device_if_available
from .helper import (
    load_jit_model,
)
MODEL_URL = "https://huggingface.co/gaspardbruno/BigLama/resolve/main/big-lama.pt"
MODEL_SHA = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"


class Lama():
    def __init__(self, device: str | None = None):
        self.device = device or get_cuda_or_mps_device_if_available()
        self.model = load_jit_model(MODEL_URL, self.device, MODEL_SHA).eval()

    def forward(self, image: torch.Tensor, mask: torch.Tensor, mode: str = 'CROP'):
        input_image = image.to(self.device)
        if image.shape[1] == 4:
            input_image = rgba_to_rgb(input_image)
        input_image = image.to(self.device)
        input_mask = mask.to(self.device)
        if mode == 'FULL':
            return self.forward_full(input_image, input_mask)
        elif mode == 'FIXED':
            return self.forward_fixed(input_image, input_mask)
        return self.forward_crop(input_image, input_mask)

    def forward_crop(self, image: torch.Tensor, mask: torch.Tensor):

        mask = (mask > 0.05).to(torch.float32)
        tar_size = mask.clone()
        d_kernel = torch.ones(13, 13).to(self.device)
        for _ in range(3):
            tar_size = M.dilation(tar_size, d_kernel)

        # Find bounding box coordinates
        _, _, y, x = torch.where(tar_size)
        y_min, y_max = y.min(), y.max()
        x_min, x_max = x.min(), x.max()

        x_padding = int(abs(x_max - x_min) * 0.35)
        y_padding = int(abs(y_max - y_min) * 0.35)

        # Apply padding
        original_x, original_y  = image.shape[2:]
        y_min = max(0, y_min - y_padding) # type: ignore
        y_max = min(original_x - 1, y_max + y_padding) # type: ignore
        x_min = max(0, x_min - x_padding) # type: ignore
        x_max = min(original_y - 1, x_max + x_padding) # type: ignore

        # Ensure crop dimensions are multiples of 64
        crop_height = ((y_max - y_min + 1) // 64) * 64
        crop_width = ((x_max - x_min + 1) // 64) * 64

        # Adjust bounding box dimensions
        y_max = y_min + crop_height - 1
        x_max = x_min + crop_width - 1

        # Crop image and mask
        cropped_image = image[:, :, y_min:y_max + 1, x_min:x_max + 1]
        cropped_mask = mask[:, :, y_min:y_max + 1, x_min:x_max + 1]

        for _ in range(3):
            cropped_mask = M.dilation(cropped_mask, d_kernel)

        result = self.model(cropped_image.to(self.device), cropped_mask.to(self.device))
        # Compose the result on top of the original image
        composed_image = image.clone()

        composed_image[:, :, y_min:y_max + 1, x_min:x_max + 1] = result

        return composed_image


    def forward_full(self, image: torch.Tensor, mask: torch.Tensor):
        # Resize image and mask with padding
        tensor_image, padding = self.resize_with_padding(image)
        tensor_mask, _ = self.resize_with_padding(mask)

        # Preprocess mask
        d_kernel = torch.ones(13, 13).to(self.device)
        for _ in range(3):
            tensor_mask = M.dilation(tensor_mask, d_kernel)
        tensor_mask = (tensor_mask > 0.05).to(torch.float32)

        # Perform inference
        result = self.model(tensor_image.to(self.device), tensor_mask.to(self.device))

        # Remove padding after inference
        result = self.remove_padding(result, padding)
        return result

    def forward_fixed(self, image: torch.Tensor, mask: torch.Tensor):
        original_shape = image.shape[2:]
        l_tensor_image = K.resize(image, (512, 512))
        l_tensor_mask = K.resize(mask, (512, 512))
        d_kernel = torch.ones(13, 13).to(self.device)
        # Preprocess mask
        l_tensor_mask = (l_tensor_mask > 0.05).to(torch.float32)
        for _ in range(3):
            l_tensor_mask = M.dilation(l_tensor_mask, d_kernel)

        l_result = self.model(l_tensor_image.to(self.device), l_tensor_mask.to(self.device))

        l_result = K.resize(l_result, original_shape)
        return l_result

    def resize_with_padding(self, image: torch.Tensor):
        # Get the larger dimension (width or height)
        image_width, image_height = image.shape[3], image.shape[2]
        max_dimension = max(image_width, image_height)

        # Calculate padding
        pad_x = max(0, (max_dimension - image_width) // 2)
        pad_y = max(0, (max_dimension - image_height) // 2)

        # Ensure that the dimensions are multiples of 64
        max_dimension = ((max_dimension - 1) // 64 + 1) * 64
        pad_x = max(0, (max_dimension - image_width) // 2)
        pad_y = max(0, (max_dimension - image_height) // 2)

        # Apply padding using kornia
        padded_image = F.pad(image, (pad_x, pad_x, pad_y, pad_y), value=0)

        # Resize using Kornia
        resized_image = K.resize(padded_image, (max_dimension, max_dimension))

        return resized_image, (pad_x, pad_y)

    def remove_padding(self, result: torch.Tensor, padding: tuple):
        pad_x, pad_y = padding
        return result[:, :, pad_y:result.shape[2] - pad_y, pad_x:result.shape[3] - pad_x]
