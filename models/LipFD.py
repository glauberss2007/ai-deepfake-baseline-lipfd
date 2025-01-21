import torch
import numpy as np
import torch.nn as nn
from .clip import clip  # CLIP Models
from .region_awareness import get_backbone  # Função para obter a arquitetura do backbone


# Classe LipFD: Responsável por utilizar CLIP e um backbone adicional
class LipFD(nn.Module):
    def __init__(self, name, num_classes=1):
        """
        Inicializa o modelo LipFD.

        Args:
            name (str): Nome do transformer ou modelo CLIP a ser carregado.
            num_classes (int): Número de classes (não utilizado diretamente nesta classe).
        """
        super(LipFD, self).__init__()

        # Convolução inicial para redimensionar as imagens
        # Converte a entrada de tamanho (1120, 1120) para (224, 224)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=5)

        # Carrega o modelo CLIP e o pré-processamento correspondente
        self.encoder, self.preprocess = clip.load(name, device="cpu")

        # Backbone adicional para processar as features extraídas
        self.backbone = get_backbone()

    def forward(self, x, feature):
        """
        Passagem direta (forward).

        Args:
            x (Tensor): Entrada do modelo.
            feature (Tensor): Features ou informações adicionais.

        Returns:
            Tensor: Saída do backbone aplicado aos dados de entrada e features.
        """
        # Ajusta os dados com o backbone usando entrada e features
        return self.backbone(x, feature)

    def get_features(self, x):
        """
        Extrai as features da imagem utilizando o modelo CLIP.

        Args:
            x (Tensor): Entrada da imagem.

        Returns:
            Tensor: Representação da imagem gerada pelo encoder CLIP.
        """
        # Redimensiona as imagens para (224, 224)
        x = self.conv1(x)

        # Extrai as features da imagem usando o modelo CLIP
        features = self.encoder.encode_image(x)
        return features


# Classe RALoss: Função de perda customizada para "Region Awareness"
class RALoss(nn.Module):
    def __init__(self):
        """
        Inicializa a classe RALoss.
        """
        super(RALoss, self).__init__()

    def forward(self, alphas_max, alphas_org):
        """
        Calcula a perda baseada em diferenças entre 'alphas_max' e 'alphas_org'.

        Args:
            alphas_max (list[Tensor]): Lista de tensors contendo os 'alphas' processados.
            alphas_org (list[Tensor]): Lista de tensors contendo os 'alphas' originais.

        Returns:
            Tensor: Valor da perda acumulada.
        """
        loss = 0.0
        batch_size = alphas_org[0].shape[0]  # Tamanho do lote

        # Itera sobre as camadas ou regiões das features
        for i in range(len(alphas_org)):
            loss_wt = 0.0  # Perda para uma camada específica

            # Itera sobre cada exemplo no batch
            for j in range(batch_size):
                # Calcula a penalidade baseada na exponencial da diferença entre alphas
                loss_wt += torch.Tensor([10]).to(alphas_max[i][j].device) / np.exp(
                    alphas_max[i][j] - alphas_org[i][j]
                )

            # Normaliza a perda pela quantidade de elementos no lote
            loss += loss_wt / batch_size

        return loss
