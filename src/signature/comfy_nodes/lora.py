import torch
import os
import subprocess
import random
from .categories import LORA_CAT
from ..img.tensor_image import TensorImage
from comfy import model_management # type: ignore
import folder_paths # type: ignore
BASE_COMFY_DIR = os.getcwd().split('custom_nodes')[0]
SIGNATURE_DIR =  os.path.dirname(os.path.realpath(__file__)).split('src')[0]
print("BASE_COMFY_DIR ------->", BASE_COMFY_DIR)
print("SIGNATURE_DIR ------->", SIGNATURE_DIR)

class SaveLoraCaptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "dataset_name": ("STRING", {"default": ""}),
                    "repeats": ("INT", {"default": 5, "min": 1}),
                    "images": ("IMAGE",),
                    "labels": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "prefix": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('folder_path',)
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = LORA_CAT

    def process(self, dataset_name: str, repeats:int, images: torch.Tensor, labels: str, prefix:str = "", suffix:str = ""):
        labels_list = labels.split('\n') if  "\n" in labels else [labels]

        root_folder = os.path.join(BASE_COMFY_DIR, "loras_datasets")
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)

        dataset_folder = os.path.join(root_folder, dataset_name)
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)

        images_folder = os.path.join(dataset_folder, f"{repeats}_{dataset_name}")
        if not os.path.exists(images_folder):
            os.mkdir(images_folder)

        tensor_images = TensorImage.from_comfy(images)
        for i, img in enumerate(tensor_images):
            TensorImage(img).save(os.path.join(images_folder, f"{dataset_name}_{i}.png"))
            # write txt label with the same name of the image
            with open(os.path.join(images_folder, f"{dataset_name}_{i}.txt"), 'w') as f:
                label = prefix + labels_list[i % len(labels_list)] + suffix
                f.write(label)
        return (dataset_folder,)


class LoraTraining:
    def __init__(self):
        self.launch_args = []
        self.ext_args = []

    @classmethod
    def INPUT_TYPES(cls):
         return {
            "required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "network_type": (["SD1.5", "SDXL"],),
            "networkmodule": (["networks.lora", "lycoris.kohya"], ),
            "networkdimension": ("INT", {"default": 32, "min":0}),
            "networkalpha": ("INT", {"default":32, "min":0}),
            "trainingresolution": ("INT", {"default":512, "step":8}),
            "data_path": ("STRING", {"default": "Insert path of image folders"}),
			"batch_size": ("INT", {"default": 1, "min":1}),
            "max_train_epoches": ("INT", {"default":10, "min":1}),
            "save_every_n_epochs": ("INT", {"default":10, "min":1}),
            "keeptokens": ("INT", {"default":0, "min":0}),
            "minSNRgamma": ("FLOAT", {"default":0, "min":0, "step":0.1}),
            "learningrateText": ("FLOAT", {"default":0.0001, "min":0, "step":0.00001}),
            "learningrateUnet": ("FLOAT", {"default":0.0001, "min":0, "step":0.00001}),
            "learningRateScheduler": (["cosine_with_restarts", "linear", "cosine", "polynomial", "constant", "constant_with_warmup"], ),
            "lrRestartCycles": ("INT", {"default":1, "min":1}),
            "optimizerType": (["AdamW8bit", "Lion8bit", "SGDNesterov8bit", "AdaFactor", "prodigy"], ),
            "output_name": ("STRING", {"default":'Desired name for LoRA.'}),
            "algorithm": (["lora","loha","lokr","ia3","dylora", "locon"], ),
            "networkDropout": ("FLOAT", {"default": 0, "step":0.1}),
            "clip_skip": ("INT", {"default":2, "min":1}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = LORA_CAT

    def process(self, ckpt_name, network_type, networkmodule, networkdimension, networkalpha, trainingresolution, data_path, batch_size, max_train_epoches, save_every_n_epochs, keeptokens, minSNRgamma, learningrateText, learningrateUnet, learningRateScheduler, lrRestartCycles, optimizerType, output_name, algorithm, networkDropout, clip_skip):
         #free memory first of all
        loadedmodels=model_management.current_loaded_models
        unloaded_model = False
        for i in range(len(loadedmodels) -1, -1, -1):
            m = loadedmodels.pop(i)
            m.model_unload()
            del m
            unloaded_model = True
        if unloaded_model:
            model_management.soft_empty_cache()

        train_data_dir = data_path.replace( "\\", "/")

        #ADVANCED parameters initialization
        is_v2_model=0
        network_moduke="networks.lora"
        network_dim=32
        network_alpha=32
        resolution = "512,512"
        keep_tokens = 0
        min_snr_gamma = 0
        unet_lr = "1e-4"
        text_encoder_lr = "1e-5"
        lr_scheduler = "cosine_with_restarts"
        lr_restart_cycles = 0
        optimizer_type = "AdamW8bit"
        algo= "lora"
        dropout = 0.0
        conv_dim = 4
        conv_alpha = 4
        output_dir = 'models/loras'
        # Learning rate | 学习率
        lr = "1e-4" # learning rate | 学习率，在分别设置下方 U-Net 和 文本编码器 的学习率时，该参数失效
        unet_lr = "1e-4" # U-Net learning rate | U-Net 学习率
        text_encoder_lr = "1e-5" # Text Encoder learning rate | 文本编码器 学习率
        lr_scheduler = "cosine_with_restarts" # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
        lr_warmup_steps = 0 # warmup steps | 学习率预热步数，lr_scheduler 为 constant 或 adafactor 时该值需要设为0。
        lr_restart_cycles = 1 # cosine_with_restarts restart cycles | 余弦退火重启次数，仅在 lr_scheduler 为 cosine_with_restarts 时起效。

        # 其他设置
        min_bucket_reso = 256 # arb min resolution | arb 最小分辨率
        max_bucket_reso = 1584 # arb max resolution | arb 最大分辨率

        save_model_as = "safetensors" # model save ext | 模型保存格式 ckpt, pt, safetensors

        network_module = networkmodule
        network_dim = networkdimension
        network_alpha = networkalpha
        resolution = f"{trainingresolution},{trainingresolution}"

        formatted_value = str(format(learningrateText, "e")).rstrip('0').rstrip()
        text_encoder_lr = ''.join(c for c in formatted_value if not (c == '0'))

        formatted_value2 = str(format(learningrateUnet, "e")).rstrip('0').rstrip()
        unet_lr = ''.join(c for c in formatted_value2 if not (c == '0'))

        keep_tokens = keeptokens
        min_snr_gamma = minSNRgamma
        lr_scheduler = learningRateScheduler
        lr_restart_cycles = lrRestartCycles
        optimizer_type = optimizerType
        algo = algorithm
        dropout = f"{networkDropout}"

        #generates a random seed
        theseed = random.randint(0, 2^32-1)

        # if multi_gpu:
        #     self.launch_args.append("--multi_gpu")

        # if lowram:
        #     self.ext_args.append("--lowram")

        self.ext_args.append(f"--clip_skip={clip_skip}")

        # if parameterization:
        #     self.ext_args.append("--v_parameterization")

        # if train_unet_only:
        #     self.ext_args.append("--network_train_unet_only")

        # if train_text_encoder_only:
        #     self.ext_args.append("--network_train_text_encoder_only")

        # if network_weights:
        #     self.ext_args.append(f"--network_weights={network_weights}")

        # if reg_data_dir:
        #     self.ext_args.append(f"--reg_data_dir={reg_data_dir}")

        if optimizer_type:
            self.ext_args.append(f"--optimizer_type={optimizer_type}")

        if optimizer_type == "DAdaptation":
            self.ext_args.append("--optimizer_args")
            self.ext_args.append("decouple=True")

        if network_module == "lycoris.kohya":
            self.ext_args.extend([
                f"--network_args",
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}",
                f"algo={algo}",
                f"dropout={dropout}"
            ])

        # if noise_offset != 0:
        #     self.ext_args.append(f"--noise_offset={noise_offset}")

        # if stop_text_encoder_training != 0:
        #     self.ext_args.append(f"--stop_text_encoder_training={stop_text_encoder_training}")

        # if save_state == 1:
        #     self.ext_args.append("--save_state")

        # if resume:
        #     self.ext_args.append(f"--resume={resume}")

        if min_snr_gamma != 0:
            self.ext_args.append(f"--min_snr_gamma={min_snr_gamma}")

        # if persistent_data_loader_workers:
        self.ext_args.append("--persistent_data_loader_workers")


        launchargs=' '.join(self.launch_args)
        extargs=' '.join(self.ext_args)

        pretrained_model = folder_paths.get_full_path("checkpoints", ckpt_name)

        #Looking for the training script.
        submodules_dir = os.path.join(SIGNATURE_DIR, 'src/signature/submodules')
        print(submodules_dir)
        sd_scripts_dir = os.path.join(submodules_dir, "sd-scripts")
        if network_type == "SD1.5":
            nodespath = os.path.join(sd_scripts_dir, "train_network.py")
        elif network_type == "SDXL":
            nodespath = os.path.join(sd_scripts_dir, "sdxl_train_network.py")

        print(nodespath)
        command = "python -m accelerate.commands.launch " + launchargs + f'--num_cpu_threads_per_process=8 "{nodespath}" --enable_bucket --pretrained_model_name_or_path={pretrained_model} --train_data_dir="{train_data_dir}" --output_dir="{output_dir}" --logging_dir="./logs" --log_prefix={output_name} --resolution={resolution} --network_module={network_module} --max_train_epochs={max_train_epoches} --learning_rate={lr} --unet_lr={unet_lr} --text_encoder_lr={text_encoder_lr} --lr_scheduler={lr_scheduler} --lr_warmup_steps={lr_warmup_steps} --lr_scheduler_num_cycles={lr_restart_cycles} --network_dim={network_dim} --network_alpha={network_alpha} --output_name={output_name} --train_batch_size={batch_size} --save_every_n_epochs={save_every_n_epochs} --mixed_precision="fp16" --save_precision="fp16" --seed={theseed} --cache_latents --prior_loss_weight=1 --max_token_length=225 --caption_extension=".txt" --save_model_as={save_model_as} --min_bucket_reso={min_bucket_reso} --max_bucket_reso={max_bucket_reso} --keep_tokens={keep_tokens} --xformers --shuffle_caption ' + extargs
        #print(command)
        subprocess.run(command, shell=True)
        print("Train finished")
        #input()
        return ()

NODE_CLASS_MAPPINGS = {
    "Lora Training": LoraTraining,
    "Save Lora Captions": SaveLoraCaptions,
}