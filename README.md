# Detecção de Deepfakes de Sincronização Labial (baseado no LipFD)

Este repositório contém a implementação de um modelo detector de inconsistência temporal áudio e visual para identificar Deepfakes. 
Está é uma abordagem recente que se concentra nas inconsistências temporais entre os movimentos labiais e o áudio em vídeos com sincronização labial.
Diferentemente de métodos que se baseiam apenas em recursos visuais ou de áudio, ou em um simples alinhamento quadro a quadro, este aproveita a correlação biológica 
inerente entre os movimentos labiais, a postura da cabeça e as características espectrais da fala.

## Características Principais

* **Abordagem Multimodal:** Utiliza informações de áudio e vídeo (frames de vídeo e espectrogramas).
* **Análise de Consistência Temporal:** Em vez de se concentrar apenas em artefatos visuais, o cerne está em identificar inconsistências na dinâmica temporal entre os movimentos labiais e os espectrogramas de áudio.
* **Extração de Recursos com Atenção Regional:** Módulo de conscientização regional ajusta dinamicamente a atenção para diferentes regiões faciais, focando em áreas cruciais como lábios, rosto e postura da cabeça.
* **Alta Precisão:** Precisão superior a 95% no conjunto de dados AVLips.

**Conjunto de Dados:**

O modelo é treinado e avaliado no conjunto de dados AVLips, um conjunto de dados audiovisuais de alta qualidade, em larga escala, especificamente projetado para a detecção de deepfakes de sincronização labial. 
Ele inclui mais de 340.000 amostras com diversos métodos de geração de deepfakes e variações de ruído/distorção adicionadas para maior robustez.

**Começando:**

1. **Pré-requisitos:** [Antes de começar, certifique-se de ter o gerenciador de pacotes `conda` instalado. Os pacotes Python necessários para este projeto estão listados em `requirements.txt`.]
2. **Instalação:**: 
```
conda create -n modelo-lip python==3.10
conda activate modelo-lip
pip install -r requirements.txt
```
3. **Preparação dos Dados:** [Baixe o Conjunto de Dados AVLips via {link}(https://drive.google.com/file/d/1fEiUo22GBSnWD7nfEwDW86Eiza-pOEJm/view?pli=1), depois extraia o arquivo baixado. Mova a pasta descompactada (AVLips) para a pasta raiz do projeto clonado. Para realizar o preprocessamento execute o script]
````
python preprocess.py
````
4. **Treinamento:** [Edite os caminhos para as listas de conjuntos de dados reais e falsos em options/base_options.py, alterando --fake_list_path e --real_list_path para os caminhos corretos dentro do seu conjunto de dados AVLips.]
````
python train.py
````
5. **Avaliação:** [Baixe os pesos do modelo treinado (ckpt.pth) e salve-os na pasta checkpoints/, depois baixe o Conjunto de Validação
 e extraia-o para a pasta datasets/val/. Certifique-se de que as pastas 0_real e 1_fake estejam devidamente posicionadas. Execute o script:]
````
python validate.py --real_list_path ./datasets/val/0_real --fake_list_path ./datasets/val/1_fake --ckpt ./checkpoints/ckpt.pth
````

**Desenvolvimento Futuro:**

Trabalhos futuros incluem expandir o LipFD para lidar com várias línguas, melhorar a robustez para diferentes técnicas de geração de deepfakes e desenvolver uma implementação em tempo real para aplicações práticas.

TODO: Buscar formas de melhorar!

## Referências
