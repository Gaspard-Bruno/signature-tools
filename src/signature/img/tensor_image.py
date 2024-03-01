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

    def __new__(cls, data, *args, **kwargs) -> 'TensorImage':
        return super().__new__(cls, cls.__format(data), *args, **kwargs) # type: ignore

    @staticmethod
    def __format(tensor: torch.Tensor) -> torch.Tensor:
        new_tensor = tensor.clone()
        if new_tensor.ndim not in [2, 3, 4]:
            raise ValueError("ImageArray must be 2D or 3D")
        if new_tensor.ndim == 2:
            # Grayscale image: 2D -> 1, 1, H, W
            new_tensor = new_tensor.unsqueeze(0).unsqueeze(0)
        elif new_tensor.ndim == 3:
            # RGB or grayscale image: 3D -> 1, C, H, W
            if new_tensor.shape[0] == 1:
                # Single-channel image: 1, H, W -> 1, 1, H, W
                new_tensor = new_tensor.unsqueeze(0)
            else:
                # RGB image: C, H, W -> 1, C, H, W
                new_tensor = new_tensor.unsqueeze(0)
        elif new_tensor.ndim == 4:
            # Batched image: 4D -> B, C, H, W
            pass  # No transformation needed for batched images

        if new_tensor.shape[1] not in [1, 3, 4]:
            raise ValueError("ImageArray must have 1, 3, or 4 channels")

        if new_tensor.dtype not in [torch.float16, torch.float32, torch.float64]:
            new_tensor = (new_tensor / 255.0).to(torch.float32)

        return new_tensor.to(K.utils.get_cuda_or_mps_device_if_available())

    def get_numpy_image(self) -> NDArray[np.float32]:
        return K.utils.tensor_to_image(self)

    @classmethod
    def from_numpy(cls, data: NDArray) -> 'TensorImage':
        new_data = K.utils.image_to_tensor(data)
        return cls(new_data)

    @classmethod
    def from_comfy(cls, data: torch.Tensor) -> 'TensorImage':
        if data.ndim == 4:
            data = data.permute(0, 3, 1, 2)
        return cls(data)

    @classmethod
    def from_local(cls, input_path: str) -> 'TensorImage':
        if not os.path.exists(input_path):
            raise ValueError(f"Local file not found: '{input_path}'")
        new_data = cls.__open_image(input_path)
        return cls(new_data)

    @classmethod
    def from_web(cls, input_url: str) -> 'TensorImage':
        result = urlparse(input_url)
        if not all([result.scheme, result.netloc]):
            raise ValueError(f"Invalid url: '{input_url}'")
        new_data = cls.__open_image(input_url)
        return cls(new_data)

    @classmethod
    def from_bytes(cls, buffer: bytes) -> 'TensorImage':
        new_data = cls.__open_image(buffer)
        return cls(new_data)

    @classmethod
    def from_base64(cls, base64_str: str) -> 'TensorImage':
        padding = '=' * (4 - len(base64_str) % 4)
        decoded_bytes = base64.b64decode(base64_str + padding)
        return cls(cls.from_bytes(decoded_bytes))

    @staticmethod
    def __open_image(data, **kwargs) -> torch.Tensor:
        new_data = iio.imread(data, **kwargs)
        new_data = K.utils.image_to_tensor(new_data)
        return new_data

    def get_comfy(self) -> torch.Tensor:
        """
        Returns:
            The tensor image in comfy format.
        """
        channels = self.shape[1]
        if channels in [3, 4]:
            # Permute the dimensions to move channels to the last dimension
            new_array = self.permute(0, 2, 3, 1)
        elif channels == 1:
            # Remove the batch dimension for grayscale images
            new_array = self.squeeze(0)
        else:
            raise ValueError("Invalid number of channels")
        return new_array.cpu()

    def save(self, output_path: str | bytes, extension: str = ".png") -> None:
        with open(output_path, "wb") as file:
            data = self.get_bytes(extension=extension)
            file.write(data)

    def get_bytes(self, extension: str = ".png") -> bytes:
        np_array = (self.get_numpy_image() * 255.0).astype(np.uint8)
        return iio.imwrite(uri="<bytes>", im=np_array, format=extension)  # type: ignore

    def get_base64(self, extension: str = ".png") -> str:
        data = self.get_bytes(extension=extension)
        return base64.b64encode(data).decode("utf-8")
