from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        # Inicializa o parser pai usando a classe BaseOptions
        parser = BaseOptions.initialize(self, parser)

        # Define os argumentos específicos para o treinamento
        parser.add_argument('--optim', type=str, default='adam', help='otimizador a ser usado [sgd, adam]')
        parser.add_argument('--loss_freq', type=int, default=100, help='frequência para exibir a perda no Tensorboard')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequência (em épocas) para salvar checkpoints')
        parser.add_argument('--train_split', type=str, default='train',
                            help='conjunto de treino, validação (val) ou teste (test)')
        parser.add_argument('--val_split', type=str, default='val',
                            help='conjunto de validação (val), treino ou teste (test)')
        parser.add_argument('--epoch', type=int, default=100, help='número total de épocas para treinamento')
        parser.add_argument('--beta1', type=float, default=0.9, help='termo de momentum para o otimizador Adam')
        parser.add_argument('--lr', type=float, default=2e-9, help='taxa de aprendizado inicial para o otimizador Adam')
        parser.add_argument('--pretrained_model', type=str, default='./checkpoints/experiment_name/model_epoch_29.pth',
                            help='modelo pré-treinado para fine-tuning, caso fine-tune seja True')
        parser.add_argument('--fine-tune', type=bool, default=True, help='flag para ativar ou desativar o fine-tuning')

        # Define o atributo interno 'isTrain' como True, indicando que este é um conjunto de opções de treinamento
        self.isTrain = True

        return parser
