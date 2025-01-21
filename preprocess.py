import os
import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from librosa import feature as audio


"""
Estrutura do dataset AVLips:
AVLips
├── 0_real
├── 1_fake
└── wav
    ├── 0_real
    └── 1_fake
"""

############ Parâmetros personalizados ##############
N_EXTRACT = 10   # Número de imagens extraídas de cada vídeo
WINDOW_LEN = 5   # Número de frames por janela extraída
MAX_SAMPLE = 100 # Número máximo de amostras para processar

audio_root = "./AVLips/wav"       # Diretório raiz dos arquivos de áudio
video_root = "./AVLips"           # Diretório raiz dos arquivos de vídeo
output_root = "./datasets/AVLips" # Diretório de saída para os arquivos processados
######################################################

# Labels indicando as pastas para dados reais e falsos
labels = [(0, "0_real"), (1, "1_fake")]

# Função para gerar e salvar um espectrograma a partir de um arquivo de áudio
def get_spectrogram(audio_file):
    data, sr = librosa.load(audio_file)  # Carrega o áudio e sua taxa de amostragem
    mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)  # Gera o espectrograma Mel
    plt.imsave("./temp/mel.png", mel)  # Salva o espectrograma como uma imagem temporária

# Função principal de processamento
def run():
    i = 0  # Contador de amostras processadas

    # Itera sobre as labels (pastas de dados reais e falsos)
    for label, dataset_name in labels:
        # Cria o diretório de saída para cada dataset, caso ainda não exista
        if not os.path.exists(dataset_name):
            os.makedirs(f"{output_root}/{dataset_name}", exist_ok=True)

        # Interrompe o processamento se atingir o número máximo de amostras
        if i == MAX_SAMPLE:
            break

        root = f"{video_root}/{dataset_name}"  # Caminho para as pastas de vídeos
        video_list = os.listdir(root)  # Lista todos os vídeos do dataset
        print(f"Processando {dataset_name}...")  # Mostra o status no console

        # Processa cada vídeo no dataset atual
        for j in tqdm(range(len(video_list))):
            v = video_list[j]

            # Carrega o vídeo usando OpenCV
            video_capture = cv2.VideoCapture(f"{root}/{v}")
            fps = video_capture.get(cv2.CAP_PROP_FPS)  # Obtém os frames por segundo (FPS)
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Obtém a quantidade total de frames

            # Seleciona os pontos de partida para extração dos frames
            frame_idx = np.linspace(
                0,
                frame_count - WINDOW_LEN - 1,
                N_EXTRACT,
                endpoint=True,
                dtype=np.uint8,
            ).tolist()
            frame_idx.sort()  # Ordena os índices para manter a sequência dos frames

            # Gera a sequência dos frames selecionados para extração
            frame_sequence = [
                i for num in frame_idx for i in range(num, num + WINDOW_LEN)
            ]

            frame_list = []  # Lista para armazenar os frames extraídos
            current_frame = 0  # Índice do frame atual

            # Lê e processa os frames do vídeo
            while current_frame <= frame_sequence[-1]:
                ret, frame = video_capture.read()  # Lê o próximo frame
                if not ret:
                    print(f"Erro ao ler o frame {v}: {current_frame}")  # Mostra erro se o frame não foi lido
                    break
                if current_frame in frame_sequence:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # Converte o frame para RGBA
                    frame_list.append(cv2.resize(frame, (500, 500)))  # Redimensiona e salva o frame
                current_frame += 1

            video_capture.release()  # Libera o arquivo de vídeo

            # Carrega o arquivo de áudio correspondente ao vídeo
            name = v.split(".")[0]  # Extrai o nome do vídeo sem a extensão
            a = f"{audio_root}/{dataset_name}/{name}.wav"  # Constrói o caminho do arquivo de áudio

            group = 0  # Índice do grupo para as janelas processadas
            get_spectrogram(a)  # Gera o espectrograma do arquivo de áudio
            mel = plt.imread("./temp/mel.png") * 255  # Carrega o espectrograma e ajusta para valores inteiros
            mel = mel.astype(np.uint8)  # Converte para inteiro (uint8)
            mapping = mel.shape[1] / frame_count  # Mapeia a largura do espectrograma para o número de frames

            # Combina o espectrograma e os frames para gerar a imagem de saída
            for i in range(len(frame_list)):
                idx = i % WINDOW_LEN
                if idx == 0:  # Processa um grupo a cada janela (WINDOW_LEN frames)
                    try:
                        begin = np.round(frame_sequence[i] * mapping)  # Índice inicial do espectrograma
                        end = np.round((frame_sequence[i] + WINDOW_LEN) * mapping)  # Índice final

                        # Recorta e redimensiona a fatia do espectrograma
                        sub_mel = cv2.resize(
                            (mel[:, int(begin) : int(end)]), (500 * WINDOW_LEN, 500)
                        )

                        # Concatena os frames e o espectrograma
                        x = np.concatenate(frame_list[i : i + WINDOW_LEN], axis=1)
                        x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)

                        # Salva a imagem final no diretório de saída
                        plt.imsave(
                            f"{output_root}/{dataset_name}/{name}_{group}.png", x
                        )

                        group = group + 1  # Incrementa o índice do grupo
                    except ValueError:
                        print(f"Erro de valor: {name}")  # Mostra erro caso ocorra
                        continue

        i += 1  # Incrementa o contador de amostras processadas

# Ponto de inicialização do script
if __name__ == "__main__":
    # Cria os diretórios de saída, caso ainda não existam
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    # Cria o diretório temporário para arquivos intermediários
    if not os.path.exists("./temp"):
        os.makedirs("./temp", exist_ok=True)

    run()  # Executa a função principal