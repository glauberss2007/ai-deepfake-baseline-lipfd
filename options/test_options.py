from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        # Chama o método 'initialize' de BaseOptions para inicializar o parser pai
        parser = BaseOptions.initialize(self, parser)

        # Adiciona argumento específico para o modo de teste
        parser.add_argument('--model_path', help='Caminho para carregar o modelo durante o teste')

        # Adiciona uma flag para usar o modo de avaliação (desabilita dropout, BN, etc.)
        parser.add_argument('--eval', action='store_true', help='Usa o modo de avaliação durante o tempo de teste.')

        # Define a flag 'isTrain' como Falso, indicando que este parser é para teste, não treino
        self.isTrain = False

        return parser
