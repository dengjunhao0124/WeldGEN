from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, \
    StableDiffusionControlNetImg2ImgPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, \
    DPMSolverMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
import torch
import gradio as gr



def infer_dream_lora():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("stable-diffusion-v1-5", torch_dtype=torch.float16, variant='fp16').to("cuda")
    pipe.load_lora_weights("run/sd15-dream_lora/")

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    control_image = Image.open("img.png").convert(
        "RGB").resize((512, 512))

    prompt = "a photo of w*"
    for i in range(10):
        image = pipe(prompt, image=control_image, num_inference_steps=25, streanth=1., guidance_scale=9).images[0]
        image.save(f"res/dream_lora/dl-{i}.png")


def infer_controlnet_i2i_dream():

    controlnet = ControlNetModel.from_pretrained(
        "controlnet",
        torch_dtype=torch.float16)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipeline.load_lora_weights("run/sd15-dream_lora/")
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

    # gr.Interface.from_pipeline(pipeline).launch()

    # control_image = load_image("D:\\Zhongyou\\stable_diffusion\\Data\\controlnet\\baseline\\c_defocus_2.png")
    control_image = Image.open("img.png").convert("RGB").resize((512, 512))
    # mask_image = Image.open("D:\\Zhongyou\\stable_diffusion\\Data\\test\\mask\\c_240115_2.0kW_5.png").resize((512, 512)).convert("RGB")

    prompt = "an image of w* with baseline defect on white background"

    generator = torch.Generator(device="cuda").manual_seed(42)
    for i in range(10):
        image = pipeline(prompt, image=control_image, control_image=control_image, generator=generator, strength=1, guidance_scale=4).images[0]
        # image.convert("L").save(f"controlnet-{i}.png")
        image.save(f"res/i2i-dream/controlnet-{i}.png")



infer_controlnet_i2i_dream()