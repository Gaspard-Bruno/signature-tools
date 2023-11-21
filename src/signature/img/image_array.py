from __future__ import annotations

import io
import os
from typing import Any
from urllib.parse import urlparse
import base64
import imageio as iio
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from ..logger import console

class ImageArray(np.ndarray):
    def __new__(cls, array: NDArray[Any]):
        new_array = cls.__format_image_array(array)
        return np.asarray(new_array).view(cls)

    @staticmethod
    def __format_image_array(array: NDArray[Any]) -> NDArray[np.uint8]:
        new_array = (array if (array.dtype == np.uint8) else array * 255).astype(np.uint8)
        if len(new_array.shape) == 2:
            new_array = np.expand_dims(new_array, -1)
        if len(new_array.shape) != 3:
            raise ValueError("ImageArray must be 2D or 3D")
        if new_array.shape[-1] not in [1, 3, 4]:
            raise ValueError("ImageArray must have 1, 3 or 4 channels")
        return new_array

    @staticmethod
    def __get_channels(data: NDArray[np.uint8]) -> int:
        if len(data.shape) == 2:
            data = np.expand_dims(data, -1)
        return data.shape[-1]

    @staticmethod
    def __open_image(data: Any, **kwargs) -> NDArray[np.uint8]:
        # kwargs.update({"pilmode": kwargs.get("mode")})
        # kwargs.pop("mode")
        data = iio.imread(data, **kwargs)
        return ImageArray.__format_image_array(data)

    @classmethod
    def from_local(cls, input_path: str, **kwargs) -> ImageArray:
        """
        It takes a path to an image file, opens it, reads the file, and then returns a new Image object

        Args:
            cls: The class object.
            input_path (_SupportsPath): The path to the image file.

        Returns:
            A new instance of the class.
        """
        try:
            if not os.path.exists(input_path):
                error = ValueError(f"Local file not found: '{input_path}'")
                console.log(error)
                raise error
            return cls(cls.__open_image(input_path, **kwargs))
        except Exception as error:
            console.log(error)
            raise error

    @classmethod
    def from_web(cls, input_url: str, **kwargs) -> ImageArray:
        """
        It takes a URL, downloads the data, and returns a `DataFrame` object

        Args:
          cls: The class object that is being created.
          input_url (str): The URL of the image to be downloaded.

        Returns:
          A new instance of the class.
        """
        try:
            result = urlparse(input_url)
            if not all([result.scheme, result.netloc]):
                error = ValueError(f"Invalid url: '{input_url}'")
                console.log(error)
                raise error
            return cls(cls.__open_image(input_url, **kwargs))
        except Exception as error:
            console.log(error)
            raise error

    @classmethod
    def from_bytes(cls, buffer: bytes, **kwargs) -> ImageArray:
        """
        It takes a buffer of bytes and returns a new instance of the class

        Args:
            cls: The class object that is being instantiated.
            buffer (bytes): the image buffer

        Returns:
            A new instance of the class.
        """
        try:
            return cls(cls.__open_image(io.BytesIO(buffer), **kwargs))
        except Exception as error:
            console.log(error)
            raise error

    @classmethod
    def from_base64(cls, base64_str: str, **kwargs) -> ImageArray:
        """
        It takes a base64 string and returns a new instance of the class

        Args:
            cls: The class object that is being instantiated.
            base64 (string): the base64 string

        Returns:
            A new instance of the class.
        """
        try:
            padding = '=' * (4 - len(base64_str) % 4)
            decoded_bytes = base64.b64decode(base64_str + padding)
            return cls(cls.from_bytes(decoded_bytes, **kwargs))
        except Exception as error:
            console.log(error)
            raise error

    @property
    def width(self) -> int:
        """
        Returns:
            The width of the image.
        """
        _, img_w = self.shape[:2]
        return img_w

    @property
    def height(self) -> int:
        """
        Returns:
            The height of the image.
        """
        img_h, _ = self.shape[:2]
        return img_h

    @property
    def channels(self) -> int:
        """
        Returns:
            The number of channels in the image.
        """
        return self.__get_channels(self)

    def from_numpy(self, array: NDArray[Any]):
        """
        It takes a numpy array and returns a new instance of the class

        Args:
            array (NDArray[Any]): The numpy array.

        Returns:
            A new instance of the class.
        """
        try:
            data = self.__format_image_array(array)
            return self.__class__(data)
        except Exception as error:
            console.log(error)
            raise error

    def save(
        self,
        output_path: str | bytes,
        extension: str = ".png",
    ):
        """
        It takes a path to a file and writes the image data to it

        Args:
            output_path (str | bytes | PathLike[str] | PathLike[bytes]): The path to the file you want to save.
            extension (str): The file extension of the image. Defaults to .png
        """
        try:
            with open(output_path, "wb") as file:
                data = self.get_bytes(extension=extension)
                file.write(data)
        except Exception as error:
            console.log(error)
            raise error

    def get_bytes(self, extension: str = ".png") -> bytes:
        return iio.imwrite(uri="<bytes>", im=self, format=extension)  # type: ignore

    def get_base64(self, extension: str = ".png") -> str:
        data = self.get_bytes(extension=extension)
        return base64.b64encode(data).decode("utf-8")

    def to_pil(self):
        """
        Convert the image to a PIL image

        Returns:
            A PIL image
        """
        data = self.get_value()
        if self.channels == 1:
            data = data[:, :, 0]
        return Image.fromarray(data)

    def get_value(self, color_space: str | None = None) -> NDArray[np.uint8]:
        """
        Return the value of the image as a numpy array, can also convert to a valid Color Space

        Args:
          color_space (Optional[str]): The color space to convert the image to "RGBA", "RGB" and "ALPHA".

        Returns:
          The numpy array of the image.
        """
        output = self.copy()

        if color_space is not None:
            color = color_space.upper()
            if color == "RGBA":
                output = self.__get_rgba()
            if color == "RGB":
                output = self.__get_rgb()
            if color == "L":
                output = self.__get_l()

        return output

    def get_float_value(self, color_space: str | None = None) -> NDArray[np.float32]:
        return (self.get_value(color_space=color_space) / 255.0).astype(np.float32)

    def __get_rgba(self) -> NDArray[np.uint8]:
        if self.channels != 4:
            error = ValueError(f"Image got {self.channels}, not possible to get RGBA")
            console.log(error)
            raise error
        return self

    def __get_rgb(self) -> NDArray[np.uint8]:
        if self.channels == 4:
            temp_img = self[:, :, :3] / 255
            temp_mask = np.expand_dims(self[:, :, -1] / 255, -1)
            bg_img = np.ones((self.width, self.height, 3)) * (1 - temp_mask)
            clean_img = (temp_img * temp_mask) + bg_img
            return (clean_img * 255).astype(np.uint8)
        if self.channels < 3:
            error = ValueError(f"Image got {self.channels}, not possible to get RGB")
            console.log(error)
            raise error
        return self

    def __get_l(self) -> NDArray[np.uint8]:
        if self.channels == 4:
            return np.expand_dims(self[:, :, -1], -1)
        if self.channels == 3:
            error = ValueError(f"Image got {self.channels}, not possible to get ALPHA")
            console.log(error)
            raise error
        return self
