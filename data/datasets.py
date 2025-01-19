import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import utils


class AVLip(Dataset):
    def __init__(self, opt):
        # Verifica se o conjunto de dados é para treinamento ou validação
        assert opt.data_label in ["train", "val"]
        self.data_label = opt.data_label

        # Carrega listas de caminhos de imagens reais e falsas
        self.real_list = utils.get_list(opt.real_list_path)
        self.fake_list = utils.get_list(opt.fake_list_path)

        # Cria um dicionário para mapear caminhos de imagem para rótulos
        self.label_dict = dict()
        for i in self.real_list:
            self.label_dict[i] = 0  # 0 para imagens reais
        for i in self.fake_list:
            self.label_dict[i] = 1  # 1 para imagens falsas

        # Combina as listas de imagens reais e falsas
        self.total_list = self.real_list + self.fake_list

    def __len__(self):
        # Retorna o número total de imagens no conjunto de dados
        return len(self.total_list)

    def __getitem__(self, idx):
        # Obtém o caminho da imagem e seu rótulo correspondente
        img_path = self.total_list[idx]
        label = self.label_dict[img_path]

        # Carrega a imagem e converte para um tensor PyTorch
        img = torch.tensor(cv2.imread(img_path), dtype=torch.float32)
        img = img.permute(2, 0, 1)  # Ajusta as dimensões do tensor

        # Normaliza a imagem
        crops = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])(img)

        # Cria recortes da imagem em diferentes escalas
        # crops[0]: 1.0x, crops[1]: 0.65x, crops[2]: 0.45x
        crops = [[transforms.Resize((224, 224))(img[:, 500:, i:i + 500]) for i in range(5)], [], []]
        crop_idx = [(28, 196), (61, 163)]
        for i in range(len(crops[0])):
            # Cria recortes de tamanho médio
            crops[1].append(transforms.Resize((224, 224))
                            (crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
            # Cria recortes pequenos
            crops[2].append(transforms.Resize((224, 224))
                            (crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))

        # Redimensiona a imagem original
        img = transforms.Resize((1120, 1120))(img)

        # Retorna a imagem original redimensionada, os recortes e o rótulo
        return img, crops, label