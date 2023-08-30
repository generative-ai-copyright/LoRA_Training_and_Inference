import os
import re
import cv2
import csv
import glob
import torch
import matplotlib.pyplot as plt
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


def inference_lora(base_model="runwayml/stable-diffusion-v1-5", lora_weights_array=None, device="cuda",
                   text_encoder=False, num_images=3, prompts=None, scale=0.5,
                   file_path=os.getcwd()):
    if lora_weights_array is None:
        lora_weights_array = []
    for lora_weights in lora_weights_array:
        if prompts is None:
            prompts = ["a person in Chinese landscape painting"]
        base_model = base_model
        pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if text_encoder:
            pipe.load_lora_weights(lora_weights, use_safetensors=False)
        else:
            pipe.unet.load_attn_procs(lora_weights, use_safetensors=False)
        pipe.to(device)
        checkingpoint = os.path.dirname(lora_weights)
        for prompt in prompts:
            for n in range(1, int(num_images) + 1):
                image = pipe(prompt, num_inference_steps=25,
                             cross_attention_kwargs={"scale": scale}).images[0]
                name = prompt + " " + str(n) + ".png"
                if not os.path.exists(os.path.join(file_path, checkingpoint, prompt)):
                    os.makedirs(os.path.join(file_path, checkingpoint, prompt))
                image.save(os.path.join(file_path, checkingpoint, prompt, name))


# # noinspection PyTypeChecker
# def test_inference(dataset="mini_harvard", checkpointing_steps=40, max_train_steps=2000):
#     data = load_dataset("imagefolder", data_dir=dataset, split="train")
#     prompts = [data[i]["caption"] for i in range(data.num_rows)]
#     for step in range(checkpointing_steps, max_train_steps, checkpointing_steps):
#         for prompt in prompts:
#             lora_path = "/checkpoint-" + str(step) + "/pytorch_lora_weights.bin"
#             inference_lora(lora_weights_array=[lora_path], num_image=10, prompts=[prompt],
#                            checkingpoint="checkpt_result-" + str(step))
#     for prompt in prompts:
#         lora_path = "pytorch_lora_weights.bin"
#         inference_lora(lora_weights_array=[lora_path], num_image=10, prompts=[prompt],
#                        checkingpoint="checkpt_result-" + str(max_train_steps))


def tilesplit(image, tile_size):
    """Splits an image into tiles (patches).

    Args:
        image (torch.Tensor): Tensor of shape [h, w, c]
        tile_size (int): Size of the patch.

    Returns:
        (torch.Tensor): Tensor of patches of shape [num_patches, tile_size, tile_size, c]
    """
    tiles = image.unfold(0, tile_size, tile_size).unfold(1, tile_size, tile_size).permute(0, 1, 3, 4, 2)
    return tiles.reshape(-1, *tiles.shape[2:])


def similarity_score(image_A, image_B, tile_size=128):
    image_A_patches = tilesplit(image_A, tile_size)
    image_B_patches = tilesplit(image_B, tile_size)
    tile_distances = ((image_A_patches - image_B_patches)**2).sum(dim=(1, 2, 3))
    return 1 / (1 + torch.sqrt(torch.max(tile_distances)))


def similarity_array_score(image_A, images_B, tile_size=128):
    similarity_array = [similarity_score(image_A, image_B, tile_size) for image_B in images_B]
    return similarity_array


def load_images(paths):
    images = []
    for path in paths:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(torch.from_numpy(image))
    return images


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


def load_weights(path):
    weights_array_path = []
    folder_pattern = os.path.join(path, 'checkpoint-*')
    folders = glob.glob(folder_pattern)
    for folder in folders:
        weights_path = os.path.join(folder, 'pytorch_model.bin')
        if os.path.exists(weights_path):
            weights_array_path.append(weights_path)
    weights_array_path.sort(key=natural_sort_key)
    return weights_array_path


def load_prompts(path):
    result = []
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            result.append(row[1])
    result.sort(key=natural_sort_key)
    return result


def get_prompt_image(path, prompt):
    result = ""
    with open(os.path.join(path, "metadata.csv")) as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            if row[1] == prompt:
                result = os.path.join(path, row[0])
    return load_images([result])[0]


def prompt_images_paths(path, prompt):
    images_paths = []
    folder_pattern = os.path.join(path, 'checkpoint-*', prompt)
    folders = glob.glob(folder_pattern)
    for folder in folders:
        result = [os.path.join(folder, file) for file in os.listdir(folder)
                  if not os.path.isdir(os.path.join(folder, file))]
        images_paths.extend(result)
    images_paths.sort(key=natural_sort_key)
    return images_paths


def moving_average(data, num_images, ckpt_step):
    return [((1 + index / num_images) * ckpt_step, sum(data[index: index + num_images]) / num_images)
            for index in range(0, len(data) - num_images + 1, num_images)]


def plot(result):
    x, y = zip(*result)
    plt.scatter(x, y)
    plt.plot(x, y)
    plt.xlabel('Number of Training Steps')
    plt.ylabel('Average Similarity Score')
    plt.title('Given Prompt(s), Average Similarity Score vs. Number of Training Steps')
    plt.show()
