from feed_forward_vqgan_clip.main import load_vqgan_model, CLIP_DIM, clamp_with_grad, synth, load_clip_model
import os, sys

import config as config

from IPython.display import Image
import torch
import clip
import torchvision
import ipywidgets as widgets

from feed_forward_vqgan_clip.download_weights import model_url, download
download("https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.1/vqgan_imagenet_f16_16384.yaml")
download("https://github.com/mehdidc/feed_forward_vqgan_clip/releases/download/0.1/vqgan_imagenet_f16_16384.ckpt")


dropdown = widgets.Dropdown(
    options=model_url.keys(),
    value='cc12m_32x1024_mlp_mixer_clip_ViTB32_256x256_v0.3.th',
    description='Model:',
    disabled=False,
    layout={'width': 'max-content'},
)
model_path = dropdown.value

device = "cuda" if torch.cuda.is_available() else "cpu"




net = torch.load(model_path, map_location="cpu").to(device)
config = net.config
vqgan_config = config.vqgan_config
vqgan_checkpoint = config.vqgan_checkpoint
clip_model = config.clip_model
clip_dim = CLIP_DIM[clip_model]
if config.get("clip_model_path"):
    assert os.path.exists(config.clip_model_path)
perceptor = load_clip_model(clip_model, path=config.get("clip_model_path")).eval().requires_grad_(False).to(device)
model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]


# Please provide a single or a list of text prompts.
# Each text prompt of the list is used to generate an independent image.
texts = [
    #"Picture of a futuristic snowy city during the night, the tree is lit with a lantern.",
    #"Castle made of chocolate",
    #"Mushroom with strange colors",
    #"a professional high quality illustration of a giraffe spider chimera. a giraffe imitating a spider. a giraffe made of spider.",
    #"bedroom from 1700",
    "a terrier with a blue sky shirt",
]
toks = clip.tokenize(texts, truncate=True)
H = perceptor.encode_text(toks.to(device)).float()
with torch.no_grad():
    z = net(H)
    z = clamp_with_grad(z, z_min.min(),
                        z_max.max())
    xr = synth(model, z)
grid = torchvision.utils.make_grid(xr.cpu(), nrow=len(xr))
pil_image = torchvision.transforms.functional.to_pil_image(grid)
pil_image


# pip install torch clip-anytorch clize einops==0.3.0 fsspec ftfy imageio kornia numpy omegaconf pandas Pillow pytorch-lightning requests taming-transformers-rom1504 tensorboard torch torchaudio torchmetrics torchvision torchtext entmax x-transformers==0.19.1