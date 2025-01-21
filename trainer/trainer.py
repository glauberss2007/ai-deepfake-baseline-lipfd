import os
import torch
import torch.nn as nn
from models import build_model, get_loss


class Trainer(nn.Module):
    def __init__(self, opt):
        # Inicializa o Trainer com os parâmetros fornecidos (opt)
        self.opt = opt
        self.total_steps = 0  # Inicializa o total de etapas como 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # Diretório para salvar os checkpoints
        self.device = (  # Define o dispositivo (GPU ou CPU)
            torch.device("cuda:{}".format(opt.gpu_ids[0]))
            if opt.gpu_ids
            else torch.device("cpu")
        )
        self.opt = opt
        self.model = build_model(opt.arch)  # Constrói o modelo com a arquitetura especificada

        # Define o step_bias dependendo se será feita fine-tune ou não
        self.step_bias = (
            0
            if not opt.fine_tune
            else int(opt.pretrained_model.split("_")[-1].split(".")[0]) + 1
        )
        # Caso seja feito fine-tune, carrega o estado do modelo pré-treinado
        if opt.fine_tune:
            state_dict = torch.load(opt.pretrained_model, map_location="cpu")  # Carrega o modelo pré-treinado
            self.model.load_state_dict(state_dict["model"])  # Carrega os pesos do modelo
            self.total_steps = state_dict["total_steps"]  # Define o total de etapas a partir do modelo salvo
            print(f"Model loaded @ {opt.pretrained_model.split('/')[-1]}")

        # Define quais parâmetros do modelo são treináveis (opcionalmente trava o encoder)
        if opt.fix_encoder:
            params = []
            for name, p in self.model.named_parameters():
                if name.split(".")[0] in ["encoder"]:  # Paralisa o treinamento para o encoder
                    p.requires_grad = False
                else:
                    p.requires_grad = False  # Paralisa o treinamento para outro bloco (parece um bug, pois é idêntico)
            params = self.model.parameters()

        # Define o otimizador, podendo ser Adam ou SGD, conforme especificado em `opt`
        if opt.optim == "adam":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim should be [adam, sgd]")  # Lança erro caso o otimizador não seja válido

        # Define as funções de perda (loss functions)
        self.criterion = get_loss().to(self.device)  # Critério de perda customizado
        self.criterion1 = nn.CrossEntropyLoss()  # Perda de entropia cruzada

        # Move o modelo para o dispositivo de treino (GPU ou CPU)
        self.model.to(opt.gpu_ids[0] if torch.cuda.is_available() else "cpu")

    # Ajusta a taxa de aprendizado, reduzindo-a por um fator de 10, até o limite mínimo
    def adjust_learning_rate(self, min_lr=1e-8):
        for param_group in self.optimizer.param_groups:  # Itera sobre os grupos de parâmetros
            if param_group["lr"] < min_lr:  # Verifica se alcançou a menor taxa de aprendizado permitida
                return False
            param_group["lr"] /= 10.0  # Reduz a taxa de aprendizado por 10
        return True

    # Define os dados de entrada para o modelo
    def set_input(self, input):
        self.input = input[0].to(self.device)  # Dados de entrada na GPU/CPU
        self.crops = [[t.to(self.device) for t in sublist] for sublist in input[1]]  # Ajusta os crops
        self.label = input[2].to(self.device).float()  # Move os rótulos para a GPU/CPU

    # Executa a passagem à frente (forward) do modelo
    def forward(self):
        self.get_features()  # Extrai as features das entradas
        self.output, self.weights_max, self.weights_org = self.model.forward(
            self.crops, self.features
        )  # Obtem as predições e pesos
        self.output = self.output.view(-1)  # Ajusta o formato da saída
        # Calcula a perda combinada das saídas
        self.loss = self.criterion(
            self.weights_max, self.weights_org
        ) + self.criterion1(self.output, self.label)

    # Retorna a perda calculada
    def get_loss(self):
        loss = self.loss.data.tolist()  # Converte a perda para lista
        return loss[0] if isinstance(loss, type(list())) else loss  # Retorna a primeira perda se for uma lista

    # Executa a retropropagação (backward) e atualiza os pesos
    def optimize_parameters(self):
        self.optimizer.zero_grad()  # Zera os gradientes
        self.loss.backward()  # Calcula os gradientes
        self.optimizer.step()  # Atualiza os pesos com o otimizador

    # Extrai as features das entradas (do modelo)
    def get_features(self):
        self.features = self.model.get_features(self.input).to(
            self.device
        )  # Extrai e move para GPU/CPU

    # Muda o modelo para o modo de avaliação
    def eval(self):
        self.model.eval()

    # Testa o modelo sem calcular gradientes
    def test(self):
        with torch.no_grad():  # Desabilita o cálculo de gradiente
            self.forward()

    # Salva os pesos da rede e o estado do otimizador
    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)  # Caminho do arquivo salvo

        # Serializa o modelo e o otimizador em um dicionário
        state_dict = {
            "model": self.model.state_dict(),  # Pesos do modelo
            "optimizer": self.optimizer.state_dict(),  # Estado do otimizador
            "total_steps": self.total_steps,  # Contador de etapas totais
        }

        torch.save(state_dict, save_path)  # Salva o dicionário no arquivo
