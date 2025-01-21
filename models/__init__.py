from .clip_models import CLIPModel
from .LipFD import LipFD, RALoss

# Lista de nomes válidos para selecionar modelos
VALID_NAMES = [
    "CLIP:ViT-B/32",  # Modelo CLIP com Vision Transformer B/32
    "CLIP:ViT-B/16",  # Modelo CLIP com Vision Transformer B/16
    "CLIP:ViT-L/14",  # Modelo CLIP com Vision Transformer L/14
]


# Função para obter um modelo pelo nome fornecido
def get_model(name):
    # Verifica se o nome do modelo está na lista de nomes válidos
    assert name in VALID_NAMES
    if name.startswith("CLIP:"):  # Verifica se o nome começa com o prefixo "CLIP:"
        # Retorna uma instância do modelo CLIP a partir do nome após "CLIP:"
        return CLIPModel(name[5:])  # Remove o prefixo "CLIP:" antes de passar o nome para a classe
    else:
        # Caso contrário, gera uma falha (nome inválido, embora o `assert` no início já impeça isso)
        assert False


# Função para construir um modelo baseado no nome do transformador fornecido
def build_model(transformer_name):
    # Verifica se o nome do transformador está na lista de nomes válidos
    assert transformer_name in VALID_NAMES
    if transformer_name.startswith("CLIP:"):  # Verifica se o nome é baseado no prefixo "CLIP:"
        # Retorna uma instância da classe LipFD passando o nome do transformador após "CLIP:"
        return LipFD(transformer_name[5:])  # Remove o prefixo "CLIP:"
    else:
        # Caso contrário, gera uma falha (nome inválido)
        assert False


# Função para obter a função de perda (Loss Function)
def get_loss():
    # Retorna uma instância da classe RALoss, usada como critério de perda
    return RALoss()
