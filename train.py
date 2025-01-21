from validate import validate  # Importa a função de validação para avaliar o desempenho do modelo
from data import create_dataloader  # Função para criar o DataLoader
from trainer.trainer import Trainer  # Classe Trainer para gerenciar treinamento e otimização
from options.train_options import TrainOptions  # Opções de treinamento (configurações via args)


# Função para configurar opções específicas para validação
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)  # Lê as opções sem imprimir no console
    val_opt.isTrain = False  # Define que não é um treino (só validação)
    val_opt.data_label = "val"  # Define o rótulo como validação
    val_opt.real_list_path = "./datasets/val/0_real"  # Caminho para os dados reais
    val_opt.fake_list_path = "./datasets/val/1_fake"  # Caminho para os dados falsos
    return val_opt  # Retorna as opções de validação


if __name__ == "__main__":
    # Lê as opções de treinamento via argumentos fornecidos na execução
    opt = TrainOptions().parse()

    # Gera as opções específicas para validação
    val_opt = get_val_opt()

    # Inicializa o treinador com base nas opções de treinamento
    model = Trainer(opt)

    # Cria os DataLoaders para os dados de treinamento e validação
    data_loader = create_dataloader(opt)  # Dados de treino
    val_loader = create_dataloader(val_opt)  # Dados de validação

    # Imprime o tamanho dos DataLoaders (número de batches)
    print("Length of data loader: %d" % (len(data_loader)))
    print("Length of val  loader: %d" % (len(val_loader)))

    # Loop principal de treinamento por 'opt.epoch' épocas
    for epoch in range(opt.epoch):
        # Define o modelo no modo de treino
        model.train()
        print("epoch: ", epoch + model.step_bias)  # 'model.step_bias' pode deslocar o contador de épocas

        # Itera sobre os batches no DataLoader de treino
        for i, (img, crops, label) in enumerate(data_loader):
            # Incrementa o contador total de steps do modelo
            model.total_steps += 1

            # Define a entrada do modelo com os dados do batch
            model.set_input((img, crops, label))

            # Executa o forward (inferência) para calcular as predições
            model.forward()

            # Calcula a perda associada ao batch atual
            loss = model.get_loss()

            # Otimiza os parâmetros do modelo com base na perda calculada
            model.optimize_parameters()

            # A cada 'opt.loss_freq' iterações, imprime o valor da perda e o step atual
            if model.total_steps % opt.loss_freq == 0:
                print(
                    "Train loss: {}\tstep: {}".format(
                        model.get_loss(), model.total_steps
                    )
                )

        # A cada 'opt.save_epoch_freq' épocas, salva o modelo em um checkpoint
        if epoch % opt.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d" % (epoch + model.step_bias))
            model.save_trainer("model_epoch_%s.pth" % (epoch + model.step_bias))

        # Define o modelo no modo de avaliação para validação
        model.eval()

        # Valida o modelo usando o DataLoader de validação e a função de validação
        ap, fpr, fnr, acc = validate(model.model, val_loader, opt.gpu_ids)

        # Imprime as métricas de validação: acurácia, precisão média, FPR e FNR
        print(
            "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
                epoch + model.step_bias, acc, ap, fpr, fnr
            )
        )
