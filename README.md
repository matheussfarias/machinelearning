# RMSProp em MNIST

O presente projeto busca analisar a eficiência do uso do algorítmo [RMSProp](https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a) no dataset do MNIST.

O [MNIST](http://yann.lecun.com/exdb/mnist/) é um dataset bastante conhecido no estudo de Machine Learning, ele é um dataset dos números de 0 a 9 escritos a mão

## mnist_dataset
Na pasta **mnist_dataset** está presente as imagens de teste e treino do conjunto de dados MNIST

## funcoes.py

No arquivo **funcoes.py** estão presentes as funções úteis utilizadas na implementação do RMSProp.

## rmsprop.py

No arquivo **rmsprop.py** está a implementação propriamente dita do algoritmo RMSProp.

## resposta.txt

O arquivo **resposta.txt** mostra o tempo que durou o treinamento da rede neural, sua porcentagem de acerto em relação ao conjunto de testes pre-estabelecido e também a matriz de confusão.

A matriz de confusão está escrita no seguinte formato:

**Matriz de Confusão**:

Linha 1 - Quantidade de imagens que foram classificadas corretamente por digito (cada digito é uma coluna).

Linha 2 - Quantidade de imagens que foram classificadas incorretamente por digito (cada digito é uma coluna).

## relatorio.pdf
No arquivo **relatorio.pdf** está contido o relatório do presente projeto, onde se explica a decisão de hiperparâmetros e como todo o projeto foi elaborado.

## apresentacao.pdf
No arquivo **apresentacao.pdf** está contido a apresentação em slides do trabalho feito.

-----
Feito por **Matheus Farias**.
