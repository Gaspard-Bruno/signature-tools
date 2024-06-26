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
            "data_path": ("STRING", {"default": "Insert path of image folders", "forceInput": True}),
            "output_name": ("STRING", {"default":'NewLoraModel'}),
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "network_type": (["SD1.5", "SDXL"],),
            "training_resolution": ("INT", {"default":512, "step":8}),
            "network_module": (["networks.lora", "lycoris.kohya"], ),
            "network_dimension": ("INT", {"default": 32, "min":0}),
            "network_alpha": ("INT", {"default":32, "min":0}),
			"batch_size": ("INT", {"default": 1, "min":1}),
            "max_train_epoches": ("INT", {"default":10, "min":1}),
            "save_every_n_epochs": ("INT", {"default":10, "min":1}),
            "keep_tokens": ("INT", {"default":0, "min":0}),
            "min_SNR_gamma": ("FLOAT", {"default":0, "min":0, "step":0.1}),
            "learning_rate_text": ("FLOAT", {"default":0.0001, "min":0, "step":0.00001}),
            "learning_rate_unet": ("FLOAT", {"default":0.0001, "min":0, "step":0.00001}),
            "learning_rate_scheduler": (["cosine_with_restarts", "linear", "cosine", "polynomial", "constant", "constant_with_warmup"], ),
            "lr_restart_cycles": ("INT", {"default":1, "min":0}),
            "optimizer_type": (["AdamW8bit", "Lion8bit", "SGDNesterov8bit", "AdaFactor", "prodigy"], ),
            "algorithm": (["lora","loha","lokr","ia3","dylora", "locon"], ),
            "network_dropout": ("FLOAT", {"default": 0, "step":0.1}),
            "clip_skip": ("INT", {"default":2, "min":1,}),
            "multi_gpu": (["false", "true"], ),
            "lowram": (["false", "true"], ),
            "train_unet_only": (["false", "true"], ),
            "train_text_encoder_only": (["false", "true"], ),
            },
        }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = LORA_CAT

    def process(self, ckpt_name, network_type, network_module, network_dimension, network_alpha, training_resolution, data_path, batch_size, max_train_epoches, save_every_n_epochs, keep_tokens, min_SNR_gamma, learning_rate_text, learning_rate_unet, learning_rate_scheduler, lr_restart_cycles, optimizer_type, output_name, algorithm, network_dropout, clip_skip, multi_gpu, lowram, train_unet_only, train_text_encoder_only):
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
        network_dim=32
        resolution = "512,512"
        min_snr_gamma = 0
        unet_lr = "1e-4"
        text_encoder_lr = "1e-5"
        lr_scheduler = "cosine_with_restarts"
        algo= "lora"
        dropout = 0.0
        conv_dim = 4
        conv_alpha = 4
        output_dir = 'models/loras'

        # Learning rate
        lr = "1e-4" # learning rate
        unet_lr = "1e-4" # U-Net learning rate | U-Net
        text_encoder_lr = "1e-5" # Text Encoder learning rate
        lr_scheduler = "cosine_with_restarts" # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
        lr_warmup_steps = 0 # warmup steps

        min_bucket_reso = 256 # arb min resolution
        max_bucket_reso = 1584 # arb max resolution

        save_model_as = "safetensors" # model save ext ckpt, pt, safetensors

        network_dim = network_dimension
        resolution = f"{training_resolution},{training_resolution}"

        formatted_value = str(format(learning_rate_text, "e")).rstrip('0').rstrip()
        text_encoder_lr = ''.join(c for c in formatted_value if not (c == '0'))

        formatted_value2 = str(format(learning_rate_unet, "e")).rstrip('0').rstrip()
        unet_lr = ''.join(c for c in formatted_value2 if not (c == '0'))

        min_snr_gamma = min_SNR_gamma
        lr_scheduler = learning_rate_scheduler
        algo = algorithm
        dropout = f"{network_dropout}"

        #generates a random seed
        theseed = random.randint(0, 2^32-1)

        if multi_gpu == "true":
            self.launch_args.append("--multi_gpu")

        if lowram == "true":
            self.ext_args.append("--lowram")

        self.ext_args.append(f"--clip_skip={clip_skip}")

        if train_unet_only == "true":
            self.ext_args.append("--network_train_unet_only")

        if train_text_encoder_only == "true":
            self.ext_args.append("--network_train_text_encoder_only")


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

        if min_snr_gamma != 0:
            self.ext_args.append(f"--min_snr_gamma={min_snr_gamma}")

        self.ext_args.append("--persistent_data_loader_workers")

        launchargs=' '.join(self.launch_args)
        extargs=' '.join(self.ext_args)

        pretrained_model = folder_paths.get_full_path("checkpoints", ckpt_name)

        #Looking for the training script.
        submodules_dir = os.path.join(SIGNATURE_DIR, 'src/signature/submodules')
        print(submodules_dir)
        sd_scripts_dir = os.path.join(submodules_dir, "sd-scripts")
        nodespath = os.path.join(sd_scripts_dir, "train_network.py")
        if network_type == "SDXL":
            nodespath = os.path.join(sd_scripts_dir, "sdxl_train_network.py")

        command = "python -m accelerate.commands.launch " + launchargs + f'--num_cpu_threads_per_process=8 "{nodespath}" --enable_bucket --pretrained_model_name_or_path={pretrained_model} --train_data_dir="{train_data_dir}" --output_dir="{output_dir}" --logging_dir="./logs" --log_prefix={output_name} --resolution={resolution} --network_module={network_module} --max_train_epochs={max_train_epoches} --learning_rate={lr} --unet_lr={unet_lr} --text_encoder_lr={text_encoder_lr} --lr_scheduler={lr_scheduler} --lr_warmup_steps={lr_warmup_steps} --lr_scheduler_num_cycles={lr_restart_cycles} --network_dim={network_dim} --network_alpha={network_alpha} --output_name={output_name} --train_batch_size={batch_size} --save_every_n_epochs={save_every_n_epochs} --mixed_precision="fp16" --save_precision="fp16" --seed={theseed} --cache_latents --prior_loss_weight=1 --max_token_length=225 --caption_extension=".txt" --save_model_as={save_model_as} --min_bucket_reso={min_bucket_reso} --max_bucket_reso={max_bucket_reso} --keep_tokens={keep_tokens} --xformers --shuffle_caption ' + extargs
        subprocess.run(command, shell=True)
        return ()

NODE_CLASS_MAPPINGS = {
    "Lora Training": LoraTraining,
    "Save Lora Captions": SaveLoraCaptions,
}
