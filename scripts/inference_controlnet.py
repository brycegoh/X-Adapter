import torch
import numpy as np
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from torch import Generator
from PIL import Image
from packaging import version

from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, PretrainedConfig, SegformerFeatureExtractor, SegformerForSemanticSegmentation

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from diffusers.utils.import_utils import is_xformers_available

from model.unet_adapter import UNet2DConditionModel
from model.adapter import Adapter_XL
from pipeline.pipeline_sd_xl_adapter_controlnet import StableDiffusionXLAdapterControlnetPipeline

from scripts.utils import str2float

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from diffusers.utils import load_image
import numpy as np

def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def inference_controlnet(args):
    device = 'cuda'
    weight_dtype = torch.float16

    controlnet_condition_scale_list = str2float(args.controlnet_condition_scale_list)
    adapter_guidance_start_list = str2float(args.adapter_guidance_start_list)
    adapter_condition_scale_list = str2float(args.adapter_condition_scale_list)

    path = args.base_path
    path_sdxl = args.sdxl_path
    path_vae_sdxl = args.path_vae_sdxl
    adapter_path = args.adapter_checkpoint

    palette = ade_palette()
    
    if args.condition_type == "seg":
        controlnet_path = "lllyasviel/sd-controlnet-seg"
        def seg(seg_img):
            feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
            model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
            original_size = seg_img.size
            inputs = feature_extractor(images=seg_img, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            seg = logits.argmax(dim=1)[0]  # Assuming batch size is 1
            seg = seg.cpu().numpy()

            # Color mapping
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color
            color_seg = color_seg.astype(np.uint8)

            # Convert to image and save
            image_seg = Image.fromarray(color_seg).resize(original_size, resample=Image.NEAREST)
            return image_seg
    else:
        raise NotImplementedError("not implemented yet")

    prompt = args.prompt
    if args.prompt_sd1_5 is None:
        prompt_sd1_5 = prompt
    else:
        prompt_sd1_5 = args.prompt_sd1_5

    if args.negative_prompt is None:
        negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    else:
        negative_prompt = args.negative_prompt

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    # load controlnet
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=weight_dtype
    )
    print('successfully load controlnet')

    if args.condition_type == "seg":
        input_image = load_image(args.input_image_path).convert("RGB")
        control_image = seg(input_image)
        control_image.save(f'{args.save_path}/{prompt[:10]}_seg_condition.png')

    # load adapter
    adapter = Adapter_XL()
    ckpt = torch.load(adapter_path)
    adapter.load_state_dict(ckpt)
    adapter.to(weight_dtype)
    print('successfully load adapter')
    # load SD1.5
    noise_scheduler_sd1_5 = DDPMScheduler.from_pretrained(
        path, subfolder="scheduler"
    )
    tokenizer_sd1_5 = CLIPTokenizer.from_pretrained(
        path, subfolder="tokenizer", revision=None, torch_dtype=weight_dtype
    )
    text_encoder_sd1_5 = CLIPTextModel.from_pretrained(
        path, subfolder="text_encoder", revision=None, torch_dtype=weight_dtype
    )
    vae_sd1_5 = AutoencoderKL.from_pretrained(
        path, subfolder="vae", revision=None, torch_dtype=weight_dtype
    )
    unet_sd1_5 = UNet2DConditionModel.from_pretrained(
        path, subfolder="unet", revision=None, torch_dtype=weight_dtype
    )
    print('successfully load SD1.5')
    # load SDXL
    tokenizer_one = AutoTokenizer.from_pretrained(
        path_sdxl, subfolder="tokenizer", revision=None, use_fast=False, torch_dtype=weight_dtype
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        path_sdxl, subfolder="tokenizer_2", revision=None, use_fast=False, torch_dtype=weight_dtype
    )
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        path_sdxl, None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        path_sdxl, None, subfolder="text_encoder_2"
    )
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(path_sdxl, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        path_sdxl, subfolder="text_encoder", revision=None, torch_dtype=weight_dtype
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        path_sdxl, subfolder="text_encoder_2", revision=None, torch_dtype=weight_dtype
    )
    vae = AutoencoderKL.from_pretrained(
        path_vae_sdxl, revision=None, torch_dtype=weight_dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        path_sdxl, subfolder="unet", revision=None, torch_dtype=weight_dtype
    )
    print('successfully load SDXL')


    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
        unet_sd1_5.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()


    with torch.inference_mode():
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        refiner.enable_model_cpu_offload()

        gen = Generator("cuda")
        gen.manual_seed(args.seed)
        pipe = StableDiffusionXLAdapterControlnetPipeline(
            vae=vae,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            unet=unet,
            scheduler=noise_scheduler,
            vae_sd1_5=vae_sd1_5,
            text_encoder_sd1_5=text_encoder_sd1_5,
            tokenizer_sd1_5=tokenizer_sd1_5,
            unet_sd1_5=unet_sd1_5,
            scheduler_sd1_5=noise_scheduler_sd1_5,
            adapter=adapter,
            controlnet=controlnet,
        )

        pipe.enable_model_cpu_offload()

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler_sd1_5 = DPMSolverMultistepScheduler.from_config(pipe.scheduler_sd1_5.config)
        pipe.scheduler_sd1_5.config.timestep_spacing = "leading"
        pipe.unet.to(device=device, dtype=torch.float16, memory_format=torch.channels_last)

        for i in range(args.iter_num):
            for controlnet_condition_scale in controlnet_condition_scale_list:
                for adapter_guidance_start in adapter_guidance_start_list:
                    for adapter_condition_scale in adapter_condition_scale_list:
                        img = \
                            pipe(prompt=prompt, negative_prompt=negative_prompt, prompt_sd1_5=prompt_sd1_5,
                                 width=1024, height=1024, height_sd1_5=512, width_sd1_5=512,
                                 image=control_image,
                                 num_inference_steps=args.num_inference_steps,
                                 num_images_per_prompt=1, generator=gen,
                                 controlnet_conditioning_scale=controlnet_condition_scale,
                                 adapter_condition_scale=adapter_condition_scale,
                                 adapter_guidance_start=adapter_guidance_start,
                                 control_guidance_start = args.control_guidance_start,
                                 control_guidance_end = args.control_guidance_end,
                                 guidance_scale = args.guidance_scale,
                                 output_type= "pil" if args.denoising_end is None else "latent",
                                 denoising_end= None if args.denoising_end is None else args.denoising_end,
                            ).images[0]
                        if args.denoising_end is not None:
                            img = refiner(
                                prompt=prompt,
                                num_inference_steps=args.num_inference_steps,
                                denoising_start=args.denoising_end,
                                image=img,
                            ).images[0]
                        img.save(
                            f"{args.save_path}/{prompt[:10]}_{i}_ccs_{controlnet_condition_scale:.2f}_ags_{adapter_guidance_start:.2f}_acs_{adapter_condition_scale:.2f}.png")

        print(f"results saved in {args.save_path}")
