import socket
import os

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

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")





    # първите 5 неща, на които снимката прилича, биват запаметени в стрингове 1 - 5
    # всеки от тези 5 стринга бива подаден на VQGAN + CLIP, които генерират снимки 1 - 5



    # снимки 1 - 5 биват пратени на клиента обратно със следния код:

    for i in range(4):
        file_name = 'picture_' + (i+1)

        file = open(file_name, 'rb')
        image_data = file.read(4096)

        while image_data:
            client_socket.send(image_data)
            image_data = file.read(4096)

        file.close()

    client_socket.close()
