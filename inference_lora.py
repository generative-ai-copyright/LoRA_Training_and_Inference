import os
from datasets import load_dataset
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


def inference_lora(base_model="runwayml/stable-diffusion-v1-5", lora_weights=None, device="cuda", text_encoder=False,
                   num_image=5, prompt="a person in Chinese landscape painting", file_name="person", scale=0.5,
                   file_path=os.getcwd()):
    base_model = base_model
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if text_encoder:
        pipe.load_lora_weights(lora_weights, use_safetensors=False)
    else:
        pipe.unet.load_attn_procs(lora_weights, use_safetensors=False)
    pipe.to(device)
    for n in range(1, int(num_image) + 1):
        image = pipe(prompt, num_inference_steps=25,
                     cross_attention_kwargs={"scale": scale}).images[0]
        name = file_name + " " + str(n) + ".png"
        if not os.path.exists(os.path.join(file_path, file_name)):
            os.makedirs(os.path.join(file_path, file_name))
        image.save(os.path.join(file_path, file_name, name))


def test_inference(dataset="mini_harvard"):
    data = load_dataset("imagefolder", data_dir=dataset, split="train")
    num_prompts = data.num_rows
    prompts = [data[i]["caption"] for i in range(num_prompts)]
    for prompt in prompts:
        inference_lora(lora_weights="pytorch_lora_weights.bin", num_image=10, prompt=prompt, file_name=(prompt))
