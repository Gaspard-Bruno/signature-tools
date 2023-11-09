from enum import Enum, unique

import cv2
import numpy as np
from numpy.typing import NDArray
from .image_array import ImageArray


# * reference: https://docs-flutter-io.firebaseapp.com/flutter/painting/BoxFit-class.html
@unique
class ResizeMode(Enum):
    """
    An enumeration of resize modes.

    This class defines the possible modes that can be used when resizing an image. It includes `ResizeMode.CONTAIN`,
    `ResizeMode.FIT`, and `ResizeMode.FILL`.

    Args:
        CONTAIN (str): The resized image should be contained within the specified dimensions, maintaining the aspect
            ratio of the original image.
        FIT (str): The resized image should fit within the specified dimensions, maintaining the aspect ratio of the
            original image. This may result in empty space within the specified dimensions.
        FILL (str): The resized image should fill the specified dimensions, maintaining the aspect ratio of the
            original image. This may result in parts of the image being cut off.
    """

    CONTAIN = "CONTAIN"
    FIT = "FIT"
    FILL = "FILL"


# * reference: https://github.com/python-pillow/Pillow/blob/43faf9c544722bc9242b88cfb578e4ab84db5a0d/src/PIL/ImageOps.py
def resize(
    image: ImageArray | NDArray[np.uint8] | NDArray[np.float32],
    size: int | tuple[int, int],
    mode: str | ResizeMode = ResizeMode.FILL,
    anti_aliasing: bool = False,
) -> ImageArray:
    """
    Resize an image to the specified dimensions.

    This function resizes an input image to the specified dimensions using the specified resize mode. The available
    resize modes are `ResizeMode.CONTAIN`, `ResizeMode.FIT`, and `ResizeMode.FILL`. If `anti_aliasing` is set to
    `True`, the resizing will be done with anti-aliasing to smooth the image.

    Args:
        image (Union[ImageArray, np.ndarray[np.uint8], np.ndarray[np.float32]]): The input image to resize.
        size (Union[int, Tuple[int, int]]): The dimensions to resize the image to. If an integer is passed, it will
            be used as the size for both the width and height.
        mode (Union[str, ResizeMode], optional): The mode to use when resizing the image. Defaults to
            `ResizeMode.FILL`.
        anti_aliasing (bool, optional): Whether or not to apply anti-aliasing when resizing the image. Defaults to
            `False`.

    Returns:
        np.ndarray[np.uint8]: The resized image.
    """
    if not isinstance(image, ImageArray):
        image = ImageArray(image)

    if isinstance(size, int):
        size = (size, size)

    if isinstance(mode, str):
        mode = ResizeMode(mode)

    width, height = size

    input_width, input_height = image.shape[:2]

    output_width = width
    output_height = height

    input_ratio = input_width / input_height
    output_ratio = width / height

    if mode is not ResizeMode.FILL and input_ratio != output_ratio:
        adjust_height_contain = mode is ResizeMode.CONTAIN and input_ratio > output_ratio
        adjust_height_fit = mode is ResizeMode.FIT and input_ratio < output_ratio

        if adjust_height_contain or adjust_height_fit:
            output_height = round(input_height / input_width * width)
        else:
            output_width = round(input_width / input_height * height)

    inter = cv2.INTER_AREA if anti_aliasing else cv2.INTER_LINEAR
    result = cv2.resize(image, (output_width, output_height), interpolation=inter)
    if mode == ResizeMode.FIT:
        h_pos = round((output_height - height) * 0.5)
        w_pos = round((output_width - width) * 0.5)

        h_offset = height + h_pos
        w_offset = width + w_pos

        result = result[h_pos:h_offset, w_pos:w_offset]

    return ImageArray(result)
