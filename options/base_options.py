import os
import argparse
import torch


class BaseOptions:
    def __init__(self):
        # Inicializa a classe com um atributo de controle
        self.initialized = False

    def initialize(self, parser):
        # Adiciona os argumentos de linha de comando ao parser
        parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14", help="veja models/__init__.py")
        parser.add_argument("--fix_backbone", default=False)  # Parâmetro para fixar o backbone do modelo
        parser.add_argument("--fix_encoder", default=True)  # Parâmetro para fixar o encoder

        # Caminhos para listas de dados reais e falsos
        parser.add_argument("--real_list_path", default="./datasets/val/0_real")
        parser.add_argument("--fake_list_path", default="./datasets/val/1_fake")
        parser.add_argument("--data_label", default="train", help="indica se é conjunto de treino ou validação")

        # Parâmetros relacionados ao treinamento
        parser.add_argument("--batch_size", type=int, default=10, help="tamanho do batch de entrada")
        parser.add_argument("--gpu_ids", type=str, default="1", help="IDs das GPUs: ex. 0, ou 0,1,2 ou use -1 para CPU")
        parser.add_argument("--name", type=str, default="experiment_name",
                            help="nome do experimento, usado para armazenar modelos e resultados")
        parser.add_argument("--num_threads", default=0, type=int, help="número de threads para carregar dados")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints",
                            help="o local onde modelos são salvos")
        parser.add_argument("--serial_batches", action="store_true",
                            help="se verdadeiro, usa ordem sequencial de imagens para batches; caso contrário, usa aleatório")

        # Marca como inicializado após adicionar os argumentos
        self.initialized = True
        return parser

    def gather_options(self):
        # Inicializa o parser apenas se ainda não foi feito
        if not self.initialized:
            # Cria um parser com formatação padrão (mostra os valores padrão nos ajudantes da linha de comando)
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)  # Adiciona os argumentos

        # Obtém apenas as opções básicas
        opt, _ = parser.parse_known_args()  # Ignora opções desconhecidas neste ponto
        self.parser = parser
        return parser.parse_args()  # Retorna os argumentos parseados completamente

    def print_options(self, opt):
        # Gera um resumo das opções escolhidas
        message = ""
        message += "----------------- Opções ---------------\n"
        for k, v in sorted(vars(opt).items()):  # Itera sobre os argumentos
            comment = ""
            default = self.parser.get_default(k)  # Obtém o valor padrão de cada argumento
            if v != default:  # Adiciona um comentário apenas se a opção estiver diferente do valor padrão
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- Fim -------------------"
        print(message)

        # Salva as opções no disco para referência futura
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)  # Diretório para armazenar os checkpoints
        os.makedirs(expr_dir, exist_ok=True)  # Garante que o diretório exista
        file_name = os.path.join(expr_dir, "opt.txt")  # Arquivo para salvar as opções
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)  # Escreve o resumo no arquivo
            opt_file.write("\n")

    def parse(self, print_options=True):
        # Coleta as opções da linha de comando
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # Flag que indica se é treino ou teste

        # Processa o sufixo de nome, se fornecido
        if opt.suffix:
            suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            opt.name = opt.name + suffix  # Adiciona o sufixo ao nome do experimento

        # Imprime as opções, caso permitido
        if print_options:
            self.print_options(opt)

        # Configura os IDs das GPUs fornecidas
        str_ids = opt.gpu_ids.split(",")  # Divide os IDs das GPUs (string para lista)
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)  # Converte de string para inteiro
            if id >= 0:  # Adiciona à lista de GPUs se for válido (maior ou igual a 0)
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:  # Define uma das GPUs como ativa, se houver mais de uma
            torch.cuda.set_device(opt.gpu_ids[0])

        # Configurações adicionais relacionadas à manipulação de dados e transformações
        opt.rz_interp = opt.rz_interp.split(",")  # Configuração de interpolação do redimensionamento
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(",")]  # Sigma de blur (converte strings para floats)
        opt.jpg_method = opt.jpg_method.split(",")  # Métodos de compactação para JPEG
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(",")]  # Qualidades JPEG (converte strings para inteiros)
        if len(opt.jpg_qual) == 2:
            # Cria um intervalo de qualidades se forem dois valores
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Não é permitido fornecer mais de dois valores para --jpg_qual.")  # Validação de erro

        self.opt = opt  # Salva as opções no objeto
        return self.opt  # Retorna as opções parseadas
