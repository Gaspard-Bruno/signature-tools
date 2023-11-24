import cv2
import torch
from .categories import PROCESSORS_CAT
from ..src.signature.img.tensor_image import TensorImage
# from ..src.signature.models.lineart_anime import LineArtAnime
from ..src.signature.models.lineart import LineArt
from ..src.signature.models.pidinet import PidiNet
from ..src.signature.models.hednet import HedNet
from kornia.color import grayscale_to_rgb, rgb_to_grayscale
from kornia.geometry.transform import resize, pyrup
import kornia as K

import numpy as np


def get_resize_resolution(image, max_resolution: int):
    scale_factor = min(max_resolution / image.shape[2], max_resolution / image.shape[1])
    return (int(image.shape[2] * scale_factor), int(image.shape[1] * scale_factor))


class CannyEdgeProcessor():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "lower_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
            "upper_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
            "resolution": ("INT", {"default": 512, "min": 0, "max": 2048}),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = PROCESSORS_CAT

    def process(self, image: torch.Tensor, lower_threshold: float, upper_threshold: float, resolution: int):
        target_resolution = get_resize_resolution(image, resolution)
        print(image.shape)
        step = TensorImage.from_comfy(image)
        print(step.shape)
        original_size = step.size
        # input_image = resize(input_image, size=target_resolution, interpolation='bilinear')
        # input_image = rgb_to_grayscale(input_image)
  
        input_image = torch.rand(1, 1, 512, 512).to(step.device)
 
        _, results = K.filters.canny(input=input_image, low_threshold=lower_threshold, high_threshold=upper_threshold)
        results = resize(results, original_size, interpolation='bilinear')
        results = grayscale_to_rgb(results)
        results = TensorImage(results).get_comfy()
        
        # for i in range(len(images)):
        #     img = images[i]
        #     print(img.shape)
        #     step = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR) # type: ignore
        #     step = cv2.Canny(image=step,threshold1=lower_threshold, threshold2=upper_threshold) # type: ignore
        #     images[i] = step



        return (results,)

# class BinaryThresholdProcessor():

#     @classmethod
#     def INPUT_TYPES(s): # type: ignore
#         return {"required": {
#             "image": ("IMAGE",),
#             "binary_threshold": ("INT", {"default": 100, "min": 0, "max": 255}),
#             "resolution": ("INT", {"default": 512, "min": 0, "max": 2048}),
#             }}
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "process"
#     CATEGORY = PROCESSORS_CAT

#     def process(self, image: torch.Tensor, binary_threshold: int, resolution: int):
#         target_resolution = get_resize_resolution(image, resolution)
#         images = helper.tensor_to_np(image)
#         results = []
#         for i in images:
#             step = cv2.resize(i, target_resolution, interpolation=cv2.INTER_LINEAR) # type: ignore
#             step = cv2.cvtColor(step, cv2.COLOR_RGB2GRAY)
#             if binary_threshold == 0 or binary_threshold == 255:
#                 _, step = cv2.threshold(step, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#             else:
#                 _, step = cv2.threshold(step, binary_threshold, 255, cv2.THRESH_BINARY_INV)
#             result = cv2.cvtColor(step, cv2.COLOR_GRAY2RGB)
#             results.append(result)

#         results = helper.np_to_tensor(results)

#         return (results,)

# class ShuffleProcessor():

#     @classmethod
#     def INPUT_TYPES(s): # type: ignore
#         return {"required": {
#             "image": ("IMAGE",),
#             "resolution": ("INT", {"default": 512, "min": 0, "max": 2048}),
#             }}
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "process"
#     CATEGORY = PROCESSORS_CAT

#     def process(self, image: torch.Tensor, resolution: int):
#         target_resolution = get_resize_resolution(image, resolution)
#         images = helper.tensor_to_np(image)
#         results = []
#         for i in images:
#             step = cv2.resize(i, target_resolution, interpolation=cv2.INTER_LINEAR) # type: ignore
#             result = self.apply(step)
#             results.append(result)

#         results = helper.np_to_tensor(results)

#         return (results,)

#     def apply(self, img, h=None, w=None, f=None):
#         H, W, _ = img.shape
#         if h is None:
#             h = H
#         if w is None:
#             w = W
#         if f is None:
#             f = 256
#         x = self.make_noise_disk(h, w, 1, f) * float(W - 1)
#         y = self.make_noise_disk(h, w, 1, f) * float(H - 1)
#         flow = np.concatenate([x, y], axis=2).astype(np.float32)
#         return cv2.remap(img, flow, None, cv2.INTER_LINEAR) # type: ignore

#     def make_noise_disk(self, H, W, C, F):
#         noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
#         noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
#         noise = noise[F: F + H, F: F + W]
#         noise -= np.min(noise)
#         noise /= np.max(noise)
#         if C == 1:
#             noise = noise[:, :, None]
#         return noise

class HedNetProcessor():

    def __init__(self):
        self.model = HedNet()

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "is_safe": ("BOOLEAN", {"default": False},),
            "resolution": ("INT", {"default": 512, "min": 0, "max": 2048}),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = PROCESSORS_CAT

    def process(self, image: torch.Tensor, is_safe: bool,  resolution: int):

        target_resolution = (resolution, resolution)
        images = TensorImage.from_comfy(image)
        original_size = (images.shape[2], images.shape[3])
        image_hed = resize(images, size=target_resolution, interpolation='bilinear')
        results = self.model.forward(image_hed, is_safe=is_safe)
        results = grayscale_to_rgb(results)
        results = resize(results, original_size, interpolation='bilinear')
        results = TensorImage(results).get_comfy()
        return (results,)

class PidiNetProcessor():

    def __init__(self):
        self.model = PidiNet()

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "is_safe": ("BOOLEAN", {"default": False},),
            "apply_filter": ("BOOLEAN", {"default": False},),
            "resolution": ("INT", {"default": 512, "min": 0, "max": 2048}),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = PROCESSORS_CAT

    def process(self, image: torch.Tensor, is_safe: bool, apply_filter: bool, resolution: int):
        target_resolution = (resolution, resolution)
        images = TensorImage.from_comfy(image)
        original_size = (images.shape[2], images.shape[3])
        images = resize(images, size=target_resolution, interpolation='bilinear')
        results = self.model.forward(images, is_safe=is_safe, apply_filter=apply_filter)
        results = grayscale_to_rgb(results)
        results = resize(results, original_size, interpolation='bilinear')

        results = TensorImage(results).get_comfy()
        return (results,)

class TileProcessor():

    def __init__(self):
        self.model = LineArt()

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": { "image": ("IMAGE",),
                             "pyrUp_iters": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}) 
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = PROCESSORS_CAT


    def process(self, image: torch.Tensor, pyrUp_iters: int):
        input_img = TensorImage.from_comfy(image)
        _, _ , height, width = input_img.shape
        step = resize(input_img, (height // (2 ** pyrUp_iters), width // (2 ** pyrUp_iters)), align_corners=False)
        for _ in range(pyrUp_iters):
            step = pyrup(step, align_corners=False)
        results = TensorImage(step).get_comfy()
        return (results,)

class LineArtProcessor():

    def __init__(self):
        self.model = LineArt()

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "mode": (['realistic', 'coarse'],),
            "resolution": ("INT", {"default": 512, "min": 0, "max": 2048}),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = PROCESSORS_CAT


    def process(self, image: torch.Tensor, mode: str, resolution: int):
        target_resolution = (resolution, resolution)
        images = TensorImage.from_comfy(image)
        original_size = (images.shape[2], images.shape[3])
        images = resize(images, size=target_resolution, interpolation='bilinear')
        results = self.model.forward(images, mode=mode)
        results = 1 - grayscale_to_rgb(results)
        results = resize(results, original_size, interpolation='bilinear')

        results = TensorImage(results).get_comfy()
        return (results,)


# class LineArtAnimeProcessor():

#     def __init__(self):
#         self.model = LineArtAnime()

#     @classmethod
#     def INPUT_TYPES(s): # type: ignore
#         return {"required": {
#             "image": ("IMAGE",),
#             "resolution": ("INT", {"default": 512, "min": 0, "max": 2048}),
#             }}
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "process"
#     CATEGORY = PROCESSORS_CAT

#     def process(self, image: torch.Tensor,  resolution: int):
#         target_resolution = (resolution, resolution)
#         images = TensorImage.from_comfy(image)
#         original_size = (images.shape[2], images.shape[3])
#         images = resize(images, size=target_resolution, interpolation='bilinear')
#         results = self.model.forward(images)
#         results = 1 - grayscale_to_rgb(results)
#         results = resize(results, original_size, interpolation='bilinear')

#         results = TensorImage(results).get_comfy()
#         return (results,)

class ScribbleHedProcessor():

    def __init__(self):
        self.model = HedNet()

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "resolution": ("INT", {"default": 512, "min": 0, "max": 2048}),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = PROCESSORS_CAT


    def nms(self, x, t, s):
        x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

        f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
        f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

        y = np.zeros_like(x)

        for f in [f1, f2, f3, f4]:
            np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

        z = np.zeros_like(y, dtype=np.uint8)
        z[y > t] = 255
        return z

    def nms2(self, x, t, s):
        x = K.filters.gaussian_blur2d(input=x, kernel_size=(1, 1), sigma=(s, s))

        f1 = torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.float32, device=x.device)
        f2 = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.float32, device=x.device)
        f3 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device=x.device)
        f4 = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32, device=x.device)

        y = torch.zeros_like(x).to(x.device)
        for f in [f1, f2, f3, f4]:
            y = torch.where(K.morphology.dilation(tensor=x, kernel=f) == x, x, y)

        z = torch.zeros_like(y, dtype=torch.float32).to(x.device)
        z[y > t] = 1.0

        return z

    def process(self, image: torch.Tensor,  resolution: int):
        target_resolution = (resolution, resolution)
        images = TensorImage.from_comfy(image)
        original_size = (images.shape[2], images.shape[3])
        images = resize(images, size=target_resolution, interpolation='bilinear')
        results = self.model.forward(images)

        results = self.nms2(results, 0.5, 3.0)
        results = resize(results, original_size, interpolation='bilinear') # type: ignore
        results = K.filters.gaussian_blur2d(input=results, kernel_size=(1, 1), sigma=(3.0, 3.0))
        results[results > 0.01] = 1.0
        results[results < 1.0] = 0.0


        results = grayscale_to_rgb(results)
        results = TensorImage(results).get_comfy()
        print(results.shape)
        return (results,)


NODE_CLASS_MAPPINGS = {
    "Hednet Processor (SoftEdge)": HedNetProcessor,
    "Pidinet Processor (SoftEdge)": PidiNetProcessor,
    "Scribble Hed Processor": ScribbleHedProcessor,
    "LineArt Processor": LineArtProcessor,
    #"LineArt Anime Processor": LineArtAnimeProcessor,
    # "Shuffle Processor": ShuffleProcessor,
    "Canny Edge Processor": CannyEdgeProcessor,
    # "Binary Threshold Processor": BinaryThresholdProcessor,
    "Tile Processor": TileProcessor,
}