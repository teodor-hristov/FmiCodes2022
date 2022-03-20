from main import load_vqgan_model, CLIP_DIM, clamp_with_grad, synth, load_clip_model
import os, sys
import config as config

from IPython.display import Image
import torch
import clip
import torchvision
import socket
import PIL.Image
import clip
import torch
from torchvision.datasets import CIFAR100

while True:

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 6666))  #ip = 127.0.0.1 , port = 6666
    server.listen()

    client_socket, client_address = server.accept()

    file = open('image_to_be_given_to_CLIP.jpg', 'wb')
    image_chunk = client_socket.recv(4096)

    while image_chunk:
        file.write(image_chunk)
        image_chunk = client_socket.recv(4096)

    file.close()


    # снимката с име "image_to_be_given_to_CLIP" се подава на първия код, който ползва CLIP за да разпознае на какво прилича снимката

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Download the dataset
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

    # Prepare the inputs
    image = PIL.Image.open('image_to_be_given_to_CLIP.jpg')
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    # sample results in arr
    results = []

    for value, index in zip(values, indices):
        print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
        if(100 * value.item() >= 30):
            results.append(f"cifar100.classes[index]:>16s")





    # първите 5 неща, на които снимката прилича, биват запаметени в стрингове 1 - 5
    # всеки от тези 5 стринга бива подаден на VQGAN + CLIP, които генерират снимки 1 - 5



    model_path = 'cc12m_32x1024_mlp_mixer_clip_ViTB32_256x256_v0.3.th'
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


    texts = ""
    for i in results:
        texts += i + " "

    toks = clip.tokenize(texts, truncate=True)
    H = perceptor.encode_text(toks.to(device)).float()
    with torch.no_grad():
        z = net(H)
        z = clamp_with_grad(z, z_min.min(),
                            z_max.max())
        xr = synth(model, z)
    grid = torchvision.utils.make_grid(xr.cpu(), nrow=len(xr))
    pil_image = torchvision.transforms.functional.to_pil_image(grid)

    pil_image.save("img1.png")




# снимката се връща на клиента


    file = open('img1.png', 'rb')
    image_data = file.read(4096)

    while image_data:
        client_socket.send(image_data)
        image_data = file.read(4096)

    file.close()
    client_socket.close()
