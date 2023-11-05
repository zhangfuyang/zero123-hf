import os
import torch
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
from diffusers.utils import load_image
from PIL import Image
import pickle
import numpy as np
from skimage.io import imread

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def load_im(path):
    img = imread(path)
    img = img.astype(np.float32) / 255.0
    mask = img[:,:,3:]
    img[:,:,:3] = img[:,:,:3] * mask + 1 - mask # white background
    img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
    return img

model_id = "kxic/zero123-165000" # zero123-105000, zero123-165000, zero123-xl

pipe = Zero1to3StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_tiling()
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")

num_images_per_prompt = 4


# test inference pipeline
# x y z, Polar angle (vertical rotation in degrees) 	Azimuth angle (horizontal rotation in degrees) 	Zoom (relative distance from center)
cond_image = load_im("./data/condition/8d8d54678f0442ff977001bac1959615/015.png")
K, azimuths, elevations, distances, cam_poses = read_pickle(os.path.join("./data/condition/8d8d54678f0442ff977001bac1959615", f'meta.pkl'))
azi_cond = azimuths[15] * 180 / np.pi
ele_cond = -elevations[15] * 180 / np.pi

# to generate first view
azi_target = 0
ele_target = -30  # zero123 uses negative elevation!!! Actually it's 30 degree

query_pose = [ele_target-ele_cond, azi_target-azi_cond, 0.0]

H, W = (256, 256)
input_images = [cond_image]
query_poses = [query_pose]

# infer pipeline, in original zero123 num_inference_steps=76
images = pipe(input_imgs=input_images, prompt_imgs=input_images, poses=query_poses, height=H, width=W,
              guidance_scale=3.0, num_images_per_prompt=num_images_per_prompt, num_inference_steps=50).images


# save imgs
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
bs = len(input_images)
i = 0
for obj in range(bs):
    for idx in range(num_images_per_prompt):
        images[i].save(os.path.join(log_dir,f"obj{obj}_{idx}.jpg"))
        i += 1

