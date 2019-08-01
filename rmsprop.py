"""
Matheus Farias - Implementação RMSProp
"""
# Importando bibliotecas úteis
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
import time
import random
import funcoes
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier



# Leitura das imagens de treino e teste e normalização dos dados entre 0 e 1
zero = io.imread("./mnist_dataset/mnist_train0.jpg", as_gray=True)/255
um = io.imread("./mnist_dataset/mnist_train1.jpg", as_gray=True)/255
dois = io.imread("./mnist_dataset/mnist_train2.jpg", as_gray=True)/255
tres = io.imread("./mnist_dataset/mnist_train3.jpg", as_gray=True)/255
quatro = io.imread("./mnist_dataset/mnist_train4.jpg", as_gray=True)/255
cinco = io.imread("./mnist_dataset/mnist_train5.jpg", as_gray=True)/255
seis = io.imread("./mnist_dataset/mnist_train6.jpg", as_gray=True)/255
sete = io.imread("./mnist_dataset/mnist_train7.jpg", as_gray=True)/255
oito = io.imread("./mnist_dataset/mnist_train8.jpg", as_gray=True)/255
nove = io.imread("./mnist_dataset/mnist_train9.jpg", as_gray=True)/255

zero_test = io.imread("./mnist_dataset/mnist_test0.jpg", as_gray=True)/255
um_test = io.imread("./mnist_dataset/mnist_test1.jpg", as_gray=True)/255
dois_test = io.imread("./mnist_dataset/mnist_test2.jpg", as_gray=True)/255
tres_test = io.imread("./mnist_dataset/mnist_test3.jpg", as_gray=True)/255
quatro_test = io.imread("./mnist_dataset/mnist_test4.jpg", as_gray=True)/255
cinco_test = io.imread("./mnist_dataset/mnist_test5.jpg", as_gray=True)/255
seis_test = io.imread("./mnist_dataset/mnist_test6.jpg", as_gray=True)/255
sete_test = io.imread("./mnist_dataset/mnist_test7.jpg", as_gray=True)/255
oito_test = io.imread("./mnist_dataset/mnist_test8.jpg", as_gray=True)/255
nove_test = io.imread("./mnist_dataset/mnist_test9.jpg", as_gray=True)/255

# Criação dos datasets de treino, teste e validação e normalização entre 0 e 1 para o uso da função de ativação sigmoide
X_zero, Y_zero = funcoes.create_trainig_set(zero, [1,0,0,0,0,0,0,0,0,0])
X_um, Y_um = funcoes.create_trainig_set(um, [0,1,0,0,0,0,0,0,0,0])
X_dois, Y_dois = funcoes.create_trainig_set(dois, [0,0,1,0,0,0,0,0,0,0])
X_tres, Y_tres = funcoes.create_trainig_set(tres, [0,0,0,1,0,0,0,0,0,0])
X_quatro, Y_quatro = funcoes.create_trainig_set(quatro, [0,0,0,0,1,0,0,0,0,0])
X_cinco, Y_cinco = funcoes.create_trainig_set(cinco, [0,0,0,0,0,1,0,0,0,0])
X_seis, Y_seis = funcoes.create_trainig_set(seis, [0,0,0,0,0,0,1,0,0,0])
X_sete, Y_sete = funcoes.create_trainig_set(sete, [0,0,0,0,0,0,0,1,0,0])
X_oito, Y_oito = funcoes.create_trainig_set(oito, [0,0,0,0,0,0,0,0,1,0])
X_nove, Y_nove = funcoes.create_trainig_set(nove, [0,0,0,0,0,0,0,0,0,1])
X_training = X_zero + X_um + X_dois + X_tres + X_quatro + X_cinco + X_seis + X_sete + X_oito + X_nove
y_training = Y_zero + Y_um + Y_dois + Y_tres + Y_quatro + Y_cinco + Y_seis + Y_sete + Y_oito + Y_nove

X_zero_test, Y_zero = funcoes.create_trainig_set(zero_test, [1,0,0,0,0,0,0,0,0,0])
X_um_test, Y_um = funcoes.create_trainig_set(um_test, [0,1,0,0,0,0,0,0,0,0])
X_dois_test, Y_dois = funcoes.create_trainig_set(dois_test, [0,0,1,0,0,0,0,0,0,0])
X_tres_test, Y_tres = funcoes.create_trainig_set(tres_test, [0,0,0,1,0,0,0,0,0,0])
X_quatro_test, Y_quatro = funcoes.create_trainig_set(quatro_test, [0,0,0,0,1,0,0,0,0,0])
X_cinco_test, Y_cinco = funcoes.create_trainig_set(cinco_test, [0,0,0,0,0,1,0,0,0,0])
X_seis_test, Y_seis = funcoes.create_trainig_set(seis_test, [0,0,0,0,0,0,1,0,0,0])
X_sete_test, Y_sete = funcoes.create_trainig_set(sete_test, [0,0,0,0,0,0,0,1,0,0])
X_oito_test, Y_oito = funcoes.create_trainig_set(oito_test, [0,0,0,0,0,0,0,0,1,0])
X_nove_test, Y_nove = funcoes.create_trainig_set(nove_test, [0,0,0,0,0,0,0,0,0,1])

# Tratamento dos dados de teste para uma única matriz x_test e y_test
x_test = X_zero_test + X_um_test + X_dois_test + X_tres_test + X_quatro_test + X_cinco_test + X_seis_test + X_sete_test + X_oito_test + X_nove_test
y_test = Y_zero + Y_um + Y_dois + Y_tres + Y_quatro + Y_cinco + Y_seis + Y_sete + Y_oito + Y_nove

#Tratamento dos dados de treino e validação (10-Folds)
X_train, X_validation, y_train, y_validation = train_test_split(X_training, y_training, test_size=0.1, random_state=0)

# Adequando os dados ao padrão da literatura
X_train = np.transpose(X_train)
X_validation = np.transpose(X_validation)
y_train = np.transpose(y_train)
y_validation = np.transpose(y_validation)
x_test = np.transpose(x_test)
y_test = np.transpose(y_test)

"""
Configurando hiperparâmetros:
    Usa-se:
        2 Camadas:
            Entrada (Camada 0) - Dimensão X_train = (784,54024)
            Hidden Layer 1     - Dimensão w1 = (784,15)
            Saída              - Dimensão w2 = (15,10)
            
            Logo, a primeira camada possui 15 neurônios
            A segunda camada possui 10 neurônios
        
        Taxa de aprendizagem:
            alpha = 5e-4
            
        Épocas: 10
        Gradiente descendente estocástico (batch=0)
"""
w1 = np.random.randn(784,15)*0.01
b1 = np.zeros((15,1))
w2 = np.random.randn(15,10)*0.01
b2 = np.zeros((10,1))

learning_rate = 5e-4

e1 = np.matrix(np.zeros((784,15)))
e2 = np.matrix(np.zeros((15,10)))
eb1 = np.matrix(np.zeros((15,1)))
eb2 = np.matrix(np.zeros((10,1)))

# Randomiza as imagens treinadas
lista = []
for i in random.sample(range(len(X_train[0,:])),len(X_train[0,:])):
    lista.append(i)

# Transforma os ndarray em matrix
w1=np.matrix(w1)
X_train=np.matrix(X_train[:,:])
y_train=np.matrix(y_train[:,:])
w2=np.matrix(w2[:,:])
b1=np.matrix(b1[:,:])
b2=np.matrix(b2[:,:])
x_test = np.matrix(x_test[:,:])
y_test = np.matrix(y_test[:,:]) 

# Inicia temporizador
start = time.time()

# Treino
for j in range(0,20):
    for i in lista:
        
        # Forward Propagation
        z1= w1.T*X_train[:,i] + b1
        a1 = funcoes.relu(z1,0)
        z2 = (w2.T*a1) + b2
        a2 = funcoes.sigmoide(z2,0)
        
        # Back Propagation
        dz2 = a2 - y_train[:,i]
        dw2 = dz2*a1.T
        dw2 = dw2.T
        db2=dz2
        dz1 = np.multiply((w2*dz2),funcoes.relu(z1,1))
        dw1 = X_train[:,i]*dz1.T
        db1=dz1

        # Erro RMSProp
        e1 = 0.9*e1 + 0.1*np.power(dw1,2)
        e2 = 0.9*e2 + 0.1*np.power(dw2,2)
        eb1 = 0.9*eb1 + 0.1*np.power(db1,2)
        eb2 = 0.9*eb2 + 0.1*np.power(db2,2)

        # Calculo da norma do erro para atualização
        normae1=np.linalg.norm(e1)
        normae2=np.linalg.norm(e2)
        normaeb1=np.linalg.norm(eb1)
        normaeb2=np.linalg.norm(eb2)
        
        # Atualização dos parâmetros segundo RMSProp
        w1 = w1 - (learning_rate/(np.power(normae1,0.5) +1e-8))*dw1
        w2 = w2 - (learning_rate/(np.power(normae2,0.5)+1e-8))*dw2
        b1 = b1 - (learning_rate/(np.power(normaeb1,0.5)+1e-8))*db1
        b2 = b2 - (learning_rate/(np.power(normaeb2,0.5)+1e-8))*db2
        
# Finaliza temporizador        
end = time.time()
        
# Inicializa contador de acertos
acerto=0

# Define matriz confusão
mcf=np.zeros(shape=(2,10))

# Teste
for i in range (np.shape(x_test)[1]):
    
    # Forward Propagation
    z1= w1.T*x_test[:,i] + b1
    a1 = funcoes.relu(z1,0)
    z2 = (w2.T*a1) + b2
    a2 = funcoes.sigmoide(z2,0)
    
    # Tratamento da saída
    fim = funcoes.resultado(a2)
    
    # Conta acerto e encontra matriz confusão
    if np.array_equal(fim,y_test[:,i]):
        if np.array_equal(np.matrix([1,0,0,0,0,0,0,0,0,0]).T,y_test[:,i]):
            mcf[0][0]+=1
        if np.array_equal(np.matrix([0,1,0,0,0,0,0,0,0,0]).T,y_test[:,i]):
            mcf[0][1]+=1
        if np.array_equal(np.matrix([0,0,1,0,0,0,0,0,0,0]).T,y_test[:,i]):
            mcf[0][2]+=1
        if np.array_equal(np.matrix([0,0,0,1,0,0,0,0,0,0]).T,y_test[:,i]):
            mcf[0][3]+=1
        if np.array_equal(np.matrix([0,0,0,0,1,0,0,0,0,0]).T,y_test[:,i]):
            mcf[0][4]+=1
        if np.array_equal(np.matrix([0,0,0,0,0,1,0,0,0,0]).T,y_test[:,i]):
            mcf[0][5]+=1
        if np.array_equal(np.matrix([0,0,0,0,0,0,1,0,0,0]).T,y_test[:,i]):
            mcf[0][6]+=1
        if np.array_equal(np.matrix([0,0,0,0,0,0,0,1,0,0]).T,y_test[:,i]):
            mcf[0][7]+=1
        if np.array_equal(np.matrix([0,0,0,0,0,0,0,0,1,0]).T,y_test[:,i]):
            mcf[0][8]+=1
        if np.array_equal(np.matrix([0,0,0,0,0,0,0,0,0,1]).T,y_test[:,i]):
            mcf[0][9]+=1
        acerto+=1
    else:
        if np.array_equal(np.matrix([1,0,0,0,0,0,0,0,0,0]).T,y_test[:,i]):
            mcf[1][0]+=1
        if np.array_equal(np.matrix([0,1,0,0,0,0,0,0,0,0]).T,y_test[:,i]):
            mcf[1][1]+=1
        if np.array_equal(np.matrix([0,0,1,0,0,0,0,0,0,0]).T,y_test[:,i]):
            mcf[1][2]+=1
        if np.array_equal(np.matrix([0,0,0,1,0,0,0,0,0,0]).T,y_test[:,i]):
            mcf[1][3]+=1
        if np.array_equal(np.matrix([0,0,0,0,1,0,0,0,0,0]).T,y_test[:,i]):
            mcf[1][4]+=1
        if np.array_equal(np.matrix([0,0,0,0,0,1,0,0,0,0]).T,y_test[:,i]):
            mcf[1][5]+=1
        if np.array_equal(np.matrix([0,0,0,0,0,0,1,0,0,0]).T,y_test[:,i]):
            mcf[1][6]+=1
        if np.array_equal(np.matrix([0,0,0,0,0,0,0,1,0,0]).T,y_test[:,i]):
            mcf[1][7]+=1
        if np.array_equal(np.matrix([0,0,0,0,0,0,0,0,1,0]).T,y_test[:,i]):
            mcf[1][8]+=1
        if np.array_equal(np.matrix([0,0,0,0,0,0,0,0,0,1]),y_test[:,i]):
            mcf[1][9]+=1


# Porcentagem de acerto no teste
porcentagem = (acerto/np.shape(x_test)[1])*100


# Tempo de treinamento em segundos
tempo = end-start

X_train = np.transpose(X_train)
X_validation = np.transpose(X_validation)
y_train = np.transpose(y_train)
y_validation = np.transpose(y_validation)
x_test = np.transpose(x_test)
y_test = np.transpose(y_test)

# Criação da rede neural e configuração dos hiperparâmetros para uso no scikit
clf = MLPClassifier(solver='lbfgs', activation='logistic', alpha=5e-4, hidden_layer_sizes=(15), random_state=1)

# Cross Validation no scikit
resultados = cross_val_predict(clf, X_validation, y_validation, cv=10)
cros_valid_resultado = accuracy_score(np.array(y_validation),resultados) * 100

X_train = np.transpose(X_train)
X_validation = np.transpose(X_validation)
y_train = np.transpose(y_train)
y_validation = np.transpose(y_validation)
x_test = np.transpose(x_test)
y_test = np.transpose(y_test)

acerto1=0
# Contagem de acertos no treino
for i in range (np.shape(X_train)[1]):
    
    # Forward Propagation
    z1= w1.T*X_train[:,i] + b1
    a1 = funcoes.relu(z1,0)
    z2 = (w2.T*a1) + b2
    a2 = funcoes.sigmoide(z2,0)
    
    # Tratamento da saída
    fim = funcoes.resultado(a2)
    
    # Conta acerto
    if np.array_equal(fim,y_train[:,i]):
        acerto1+=1

porcentagemtreino = (acerto1/np.shape(X_train)[1])*100
# Criando e escrevendo em arquivos de texto
arquivo = open('resposta.txt','w')
arquivo.write("O tempo de treinamento total foi {0:.2f} segundos\n\n".format(tempo))
arquivo.write("A porcentagem de acertos em relação ao conjunto de teste foi {0:.2f}%\n\n".format(porcentagem))
arquivo.write("A porcentagem de acertos em relação ao conjunto de treino foi {0:.2f}%\n\n".format(porcentagemtreino))
arquivo.write("A porcentagem de acertos em relação ao conjunto de validação Cross-Validation 10 Folds foi {0:.2f}%\n\n".format(cros_valid_resultado))
arquivo.write("Matriz de Confusão:\n")
arquivo.write("{}  {}  {}  {}  {}  {}  {}  {}  {}  {}  \n".format(mcf[0][0],mcf[0][1],mcf[0][2],mcf[0][3],mcf[0][4],mcf[0][5],mcf[0][6],mcf[0][7],mcf[0][8],mcf[0][9],))
arquivo.write("{}   {}    {}  {}  {}   {}  {}   {}  {}   {}  \n".format(mcf[1][0],mcf[1][1],mcf[1][2],mcf[1][3],mcf[1][4],mcf[1][5],mcf[1][6],mcf[1][7],mcf[1][8],mcf[1][9],))
arquivo.close()