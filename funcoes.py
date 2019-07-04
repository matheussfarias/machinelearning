"""
Matheus Farias - Funções
"""
# Importando bibliotecas úteis
import numpy as np

# Função que cria dataset de treino
def create_trainig_set(image, output):
    w = int(image.shape[0]/28)
    z = int(image.shape[1]/28)
    
    X = []
    Y = []
    
    for k in range(w):
        n = 0
        for n in range(z):
            digito = image[(28*k):(28*(k+1)), (28*n):(28*(n+1))]
            if np.sum(digito)!=0:
                X.append(np.reshape(digito, 784))
                Y.append(output)        
    return X, Y

# Função de ativação ReLU
def relu(x,derivada=0):
    if type(x)==np.matrix:
        if derivada == 0:
            N=np.size(x)
            r=np.zeros_like(x)
            
            for i in range(N):
                r[i]=np.max(x[i],0)
            return r
        else:
            return np.where(x <=0, 0, 1)
    else:
        print("Not a fuck vector!!")

# Função de ativação Sigmoide
def sigmoide(x,derivada):
    if derivada ==0:
        x0= np.exp(-x)
        return 1/(1+x0)
    else:
        return sigmoide(x,0)*(1-sigmoide(x,0))
    
# Função que ajusta a saída para o formato adequado
def resultado(x):
    x = np.array(x)
    for caixa in range(0,len(x)):
        if x[caixa]==max(x):
            x[caixa] = 1;
        else:
            x[caixa]=0
    return np.matrix(x[:,:])


