import torch
import kornia
from torch.nn import functional as F

from .helper import (
    load_jit_model,
)
LAMA_MODEL_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
LAMA_MODEL_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"

class Lama():
    def __init__(self, device: str|None = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_jit_model(LAMA_MODEL_URL, self.device, LAMA_MODEL_MD5).eval()

    def forward(self, image: torch.Tensor, mask: torch.Tensor):

        # Resize image and mask with padding
        resized_image, padding = self.resize_with_padding(image)
        resized_mask, _ = self.resize_with_padding(mask)

        # Preprocess mask and move to device
        input_mask = (resized_mask > 0).float()
        input_mask = input_mask.transpose(2, 1).unsqueeze(0)

        # Move resized image to device
        input_image = resized_image.unsqueeze(0)
        print(input_image.shape, input_mask.shape)
        # Perform inference
        result = self.model(input_image.to(self.device),
                            input_mask.to(self.device))

        # Remove padding after inference
        result = self.remove_padding(result, padding).to('cpu')

        return result

    def resize_with_padding(self, image: torch.Tensor):
        # Get the larger dimension (width or height)
        max_dimension = max(image.shape[1], image.shape[2])

        # Calculate padding
        pad_x = max(0, (max_dimension - image.shape[2]) // 2)
        pad_y = max(0, (max_dimension - image.shape[1]) // 2)

        # Apply padding using kornia
        padded_image = F.pad(image, (pad_x, pad_x, pad_y, pad_y), value=0)

        # Resize using Kornia
        resized_image = kornia.geometry.transform.resize(padded_image, (max_dimension, max_dimension))

        return resized_image, (pad_x, pad_y)

    def remove_padding(self, result: torch.Tensor, padding: tuple):
        pad_x, pad_y = padding
        return result[:, :, pad_y:result.shape[2] - pad_y, pad_x:result.shape[3] - pad_x]
