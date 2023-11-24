import os
import io
import torch
import base64
import kornia as K
import numpy as np
import imageio as iio

from numpy.typing import NDArray
from urllib.parse import urlparse


class TensorImage(torch.Tensor):

    def __init__(self, data):
        self.original_data = data

    def __new__(cls, data, *args, **kwargs):
        new_data = cls.__format(data)
        return super().__new__(cls, new_data, *args, **kwargs) # type: ignore

    @staticmethod
    def __format(tensor: torch.Tensor) -> torch.Tensor:
        new_tensor = tensor.clone()
        if new_tensor.ndim not in [2, 3, 4]:
            raise ValueError("ImageArray must be 2D or 3D")
        if new_tensor.ndim == 2:
            # _, _, H, W -> 1, C, H, W
            new_tensor = new_tensor.unsqueeze(0).unsqueeze(0)
        if new_tensor.ndim == 3:
            # _, H, W -> 1, C, H, W
            new_tensor = new_tensor.unsqueeze(0)

        if new_tensor.shape[1] not in [1, 3, 4]:
            raise ValueError("ImageArray must have 1, 3 or 4 channels")

        if new_tensor.dtype not in [torch.float16, torch.float32, torch.float64]:
            new_tensor = (new_tensor / 255.0).to(torch.float32)
        device = K.utils.get_cuda_or_mps_device_if_available()
        return new_tensor.to(device)

    def get_numpy_image(self) -> NDArray[np.float32]:
        return K.utils.tensor_to_image(self)

    @classmethod
    def from_numpy(cls, data: NDArray):
        new_data = K.utils.image_to_tensor(data)
        return cls(new_data)
    
    @classmethod
    def from_comfy(cls, data: torch.Tensor):
        if data.ndim == 4:
            data = data.permute(0, 3, 1, 2)
        return cls(data)

    @classmethod
    def from_local(cls, input_path: str):
        if not os.path.exists(input_path):
            raise ValueError(f"Local file not found: '{input_path}'")
        new_data = cls.__open_image(input_path)
        return cls(new_data)

    @classmethod
    def from_web(cls, input_url: str):
        result = urlparse(input_url)
        if not all([result.scheme, result.netloc]):
            raise ValueError(f"Invalid url: '{input_url}'")
        new_data = cls.__open_image(input_url)
        return cls(new_data)

    @classmethod
    def from_bytes(cls, buffer: bytes):
        b_buffer = io.BytesIO(buffer)
        new_data = cls.__open_image(b_buffer)
        return cls(new_data)

    @classmethod
    def from_base64(cls, base64_str: str):
        padding = '=' * (4 - len(base64_str) % 4)
        decoded_bytes = base64.b64decode(base64_str + padding)
        return cls(cls.from_bytes(decoded_bytes))

    @staticmethod
    def __open_image(data, **kwargs) -> torch.Tensor:
        new_data = iio.imread(data, **kwargs)
        new_data = K.utils.image_to_tensor(new_data)
        return new_data

    @property
    def size(self) -> tuple[int, int]:
        """
        Returns:
            The size of the tensor image.
        """
        return (self.height, self.width)
    @property
    def width(self) -> int:
        """
        Returns:
            The width of the tensor image.
        """
        return self.shape[3]

    @property
    def height(self) -> int:
        """
        Returns:
            The height of the tensor image.
        """
        return self.shape[2]

    @property
    def channels(self) -> int:
        """
        Returns:
            The number of channels of the tensor image.
        """
        return self.shape[1]
    @property
    def batch_size(self) -> int:
        """
        Returns:
            The number of batches of the tensor image.
        """
        return self.shape[0]

    def get_comfy(self) -> torch.Tensor:
        """
        Returns:
            The tensor image in comfy format.
        """
        if self.channels in [3, 4]:
            new_array = self.permute(0, 2, 3, 1)
        elif self.channels == 1:
            new_array = self.squeeze(0)
        else:
            raise ValueError("Invalid number of channels")
        return new_array.cpu()

    def save(self,output_path: str | bytes, extension: str = ".png"):
        with open(output_path, "wb") as file:
            data = self.get_bytes(extension=extension)
            file.write(data)

    def get_bytes(self, extension: str = ".png") -> bytes:
        np_array = (self.get_numpy_image() * 255.0).astype(np.uint8)
        return iio.imwrite(uri="<bytes>", im=np_array, format=extension)  # type: ignore

    def get_base64(self, extension: str = ".png") -> str:
        data = self.get_bytes(extension=extension)
        return base64.b64encode(data).decode("utf-8")