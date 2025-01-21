import os

def get_list(path) -> list:
    r"""Recursively read all files in root path"""
    # Lista para armazenar os caminhos completos das imagens encontradas
    image_list = list()

    # Caminha recursivamente pelo diretório fornecido
    for root, dirs, files in os.walk(path):
        # Itera sobre todos os arquivos na pasta atual
        for f in files:
            # Verifica se a extensão do arquivo é de uma imagem válida (png, jpg, jpeg)
            if f.split('.')[1] in ['png', 'jpg', 'jpeg']:
                # Adiciona o caminho completo do arquivo à lista de imagens
                image_list.append(os.path.join(root, f))

    # Retorna a lista de caminhos completos das imagens encontradas
    return image_list
