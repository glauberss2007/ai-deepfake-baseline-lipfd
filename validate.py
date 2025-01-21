import argparse
import torch
import numpy as np
from data import AVLip
import torch.utils.data
from models import build_model
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score

# Função para validar o modelo usando um conjunto de dados fornecido
def validate(model, loader, gpu_id):
    print("validating...")  # Indica que o processo de validação começou

    # Define o dispositivo onde o cálculo será realizado (GPU ou CPU)
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():  # Desabilita o cálculo do gradiente para economizar memória e acelerar o processo
        y_true, y_pred = [], []  # Inicializa listas para armazenar os rótulos verdadeiros e previstos

        # Itera sobre os lotes (batches) de dados no carregador
        for img, crops, label in loader:
            # Transfere as imagens e os crops para o dispositivo
            img_tens = img.to(device)
            crops_tens = [[t.to(device) for t in sublist] for sublist in crops]

            # Extrai as características das imagens através do modelo
            features = model.get_features(img_tens).to(device)

            # Realiza a predição e aplica a sigmoide para obter probabilidades
            y_pred.extend(model(crops_tens, features)[0].sigmoid().flatten().tolist())

            # Armazena os rótulos verdadeiros
            y_true.extend(label.flatten().tolist())

    # Converte as listas de rótulos em arrays NumPy para facilitar os cálculos métricos
    y_true = np.array(y_true)
    y_pred = np.where(np.array(y_pred) >= 0.5, 1, 0)  # Aplica um limiar para decidir entre os rótulos 0 e 1

    # Calcula a métrica de precisão média (Average Precision)
    ap = average_precision_score(y_true, y_pred)

    # Calcula a matriz de confusão (true positive, false negative, false positive, true negative)
    cm = confusion_matrix(y_true, y_pred)
    tp, fn, fp, tn = cm.ravel()  # Extrai os valores diretamente da matriz de confusão

    # Calcula a taxa de falsos negativos (FNR) e a taxa de falsos positivos (FPR)
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)

    # Calcula a acurácia do modelo
    acc = accuracy_score(y_true, y_pred)

    return ap, fpr, fnr, acc  # Retorna as métricas


# Se o script for executado diretamente (e não importado), ele executa o código abaixo
if __name__ == "__main__":
    # Configuração dos argumentos que podem ser passados via linha de comando
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Caminho para o conjunto de dados real
    parser.add_argument("--real_list_path", type=str, default="./datasets/val/0_real")

    # Caminho para o conjunto de dados falso
    parser.add_argument("--fake_list_path", type=str, default="./datasets/val/1_fake")

    # Número máximo de amostras para uso na validação
    parser.add_argument("--max_sample", type=int, default=1000, help="max number of validate samples")

    # Tamanho dos lotes de dados
    parser.add_argument("--batch_size", type=int, default=10)

    # Nome do rótulo dos dados (validação)
    parser.add_argument("--data_label", type=str, default="val")

    # Arquitetura do modelo a ser usada
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")

    # Caminho para o arquivo de checkpoints do modelo
    parser.add_argument("--ckpt", type=str, default="./checkpoints/ckpt.pth")

    # ID da GPU a ser usado
    parser.add_argument("--gpu", type=int, default=0)

    # Analisa os argumentos passados pela linha de comando
    opt = parser.parse_args()

    # Definição do dispositivo para execução (GPU específica ou CPU)
    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using cuda {opt.gpu} for inference.")  # Imprime o dispositivo em uso

    # Constrói o modelo usando a arquitetura fornecida
    model = build_model(opt.arch)

    # Carrega os pesos do modelo (checkpoint fornecido)
    state_dict = torch.load(opt.ckpt, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    print("Model loaded.")  # Confirma que o modelo foi carregado

    # Define o modelo para o modo de avaliação
    model.eval()
    model.to(device)  # Move o modelo para o dispositivo definido

    # Define o conjunto de dados e inicializa o carregador (DataLoader) para iterar sobre os dados
    dataset = AVLip(opt)
    loader = data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True
    )

    # Valida o modelo e calcula as métricas de desempenho
    ap, fpr, fnr, acc = validate(model, loader, gpu_id=[opt.gpu])

    # Imprime as métricas de desempenho
    print(f"acc: {acc} ap: {ap} fpr: {fpr} fnr: {fnr}")
