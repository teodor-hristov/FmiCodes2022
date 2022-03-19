import socket

while True:

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind('localhost', 6666)  #ip = 127.0.0.1 , port = 6666
    server.listen()

    client_socket, client_address = server.accept()

    file = open('image_to_be_given_to_CLIP', 'wb')
    image_chunk = client_socket.recv(4096)

    while image_chunk:
        file.write(image_chunk)
        image_chunk = client_socket.recv(4096)

    file.close()


    # снимката с име "image_to_be_given_to_CLIP" се подава на първия код, който ползва CLIP за да разпознае на какво прилича снимката
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
