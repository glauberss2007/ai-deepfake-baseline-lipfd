from .clip import clip  # Importa o módulo CLIP
from PIL import Image  # Biblioteca para manipulação de imagens
import torch.nn as nn  # Biblioteca do PyTorch para criar redes neurais

# Dicionário para mapear os modelos CLIP aos seus respectivos tamanhos de saída (número de canais)
CHANNELS = {
    "RN50": 1024,  # ResNet50: saída com 1024 canais
    "ViT-L/14": 768  # Vision Transformer (ViT-L/14): saída com 768 canais
}


# Definição da classe CLIPModel que herda de nn.Module
class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        """
        Inicializa o modelo CLIP.

        Args:
            name (str): Nome do modelo (deve estar em CHANNELS, ex. 'ViT-L/14' ou 'RN50').
            num_classes (int): Número de classes para a tarefa de classificação. Por padrão, 1 (tarefa binária).
        """
        super(CLIPModel, self).__init__()

        # Carrega o modelo CLIP e a função de pré-processamento
        self.model, self.preprocess = clip.load(name,
                                                device="cpu")  # self.preprocess não será usada durante o treinamento.

        # Camada fc (fully connected) para ajustar a saída do modelo ao número de classes
        self.fc = nn.Linear(CHANNELS[name],
                            num_classes)  # Entrada da camada totalmente conectada depende do modelo escolhido

    def forward(self, x, return_feature=False):
        """
        Define a passagem de dados pelo modelo no modo 'forward'.

        Args:
            x (Tensor): Entrada no formato esperado pelo modelo CLIP.
            return_feature (bool): Se for True, retorna as features (saídas intermediárias do CLIP).

        Returns:
            Tensor: Se return_feature=True, retorna as features; caso contrário, retorna as previsões da camada fully connected (fc).
        """
        # Extrai as features do modelo (aplicação da codificação de imagem do CLIP)
        features = self.model.encode_image(x)

        # Se return_feature for True, retorna apenas as features intermediárias
        if return_feature:
            return features

        # Caso contrário, aplica a camada fully connected (fc) para obter a previsão
        return self.fc(features)
