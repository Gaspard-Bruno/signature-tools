# import torch
# from kornia.utils import get_cuda_or_mps_device_if_available
# import kornia.geometry.transform as K
# import torchvision.transforms.functional as T
# from .helper import (
#     load_jit_model,
# )

# MODEL_URL = "/resources/repos/ComfyUI/custom_nodes/signature/fba_matting.pt"
# MODEL_SHA = ""

# class FbaMatting():
#     def __init__(self):
#         self.device = get_cuda_or_mps_device_if_available()
#         self.model = load_jit_model(MODEL_URL, self.device, MODEL_SHA).eval().to(self.device)

#     def forward(self, image: torch.Tensor, trimap: torch.Tensor):
#         _,_,H, W = image.shape

#         infer_size = ((max(H, W) - 1) // 8 + 1) * 8
#         resized_image = T.resize(image, size=[infer_size, infer_size], interpolation=T.InterpolationMode.BICUBIC)
#         resized_trimap = T.resize(trimap, size=[infer_size, infer_size], interpolation=T.InterpolationMode.BICUBIC)
#         # resized_image = K.resize(image, (infer_size, infer_size), interpolation='bilinear')
#         # resized_trimap = K.resize(trimap, (infer_size, infer_size), interpolation='bilinear')

#         result = self.model(resized_image, resized_trimap)[0]
#         pred = K.resize(result[0], (H, W), interpolation='bilinear')

#         return pred
