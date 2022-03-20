import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1', 6666))  #ip = 127.0.0.1 , port = 6666

file = open('dogo.jpg', 'rb')
image_data = file.read(4096)

while image_data:
    client.send(image_data)
    image_data = file.read(4096)

file.close()
client.close()
