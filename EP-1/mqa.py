import pandas as pd
import numpy as np
import plotly.graph_objects as go
import chart_studio.plotly as py
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import datasets

PATH = "iris.data"

names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "Class"]

data = pd.read_csv(PATH, names=names)

# Na linha abaixo, estamos dando um DROP nas colunas que não são utilizadas
data.dropna(how="all", inplace=True)

# Nas linhas 20 e 21, estamos dividindo nosso dataset. Em X fica a matriz das features e em y ficam os labels
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

# Na linha abaixo estamos normalizando nosso dataset
sc = StandardScaler()
X_std = sc.fit_transform(X)

# Na linha abaixo criamos nosso vetor de médias
vetor_das_medias = np.mean(X_std, axis=0)

# Na linha abaixo estamos criando a nossa matriz de covariância
matriz_cov = np.cov(X_std.T)

# Na linha abaixo estamos printando a matriz de covariância
print("\nMatriz de Covariância \n", matriz_cov)

# Na linha abaixo estamos criando os nossos autovalores e autovetores utilizando a matriz de Covariância
autovalores_cov, autovetores_cov = np.linalg.eig(matriz_cov)

# Autovetores e Autovalores com matriz de Covariância
print("\n---------Com Matriz de Covariância---------")

# Nas linhas 43 e 44 estamos printando nossos Autovetores e Autovalores
print("\nAutovetores \n", autovetores_cov)
print("\nAutovalores \n", autovalores_cov)

# Na linha abaixo estamos criando nossa matriz de Correlação
matriz_correlacao = np.corrcoef(X_std.T)

# Na linha abaixo estamos criando os nossos autovalores e autovetores utilizando a matriz de Correlação
autovalores_correlacao, autovetores_correlacao = np.linalg.eig(matriz_correlacao)

# Autovetores e Autovalores com matriz de Correlação
print("\n---------Com Matriz de Covariância---------")

# Nas linhas 56 e 57, estamos printando os Autovetores e Autovalores
print("\nAutovetores \n", autovetores_correlacao)
print("\nAutovalores \n", autovalores_correlacao)

for ev in autovetores_cov:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

# Na linha abaixo, estamos fazendo uma lista de tuplas com (autovalor, autovetor)
pares = [
    (np.abs(autovalores_cov[i]), autovetores_cov[:, i])
    for i in range(len(autovalores_cov))
]

# Nas linhas 66 e 67, estamos ordenando nossas tuplas decrescentemente
pares.sort()
pares.reverse()

# Nas linhas 70-72, estamos confirmando que nossa lista foi ordenada corretamente
print("Autovalores em ordem decrescente")
for i in pares:
    print(i[0])

# Nas linhas 76-77, estamos criando a nossa explicacao da variancia dividindo cada valor dos autovetores de correlação pela soma dos autovalores
tot = sum(autovalores_correlacao)
explicacao_variancia = [
    (i / tot) * 100 for i in sorted(autovalores_correlacao, reverse=True)
]

#Nas linhas 81-87 estamos criando o gráfico da Explicação da Variância
obj = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
plt.bar(obj, explicacao_variancia, color="blue")
plt.xticks(obj)
plt.ylabel('Pocentagem')
plt.xlabel('Features')
plt.title("Explicação da Variância")
plt.figure()

# Na linha abaixo, estamos printando a explicacao da variancia de cada feature
print("\nExplicacao das variancias\n",explicacao_variancia)

#Na linha abaixo estamos reduzindo a nossa matriz para dimensao 2
matriz_reduzida = np.hstack((pares[0][1].reshape(4, 1), pares[1][1].reshape(4, 1)))

#Na linha abaixo estamos printando a nossa matriz reduzida
print("\nMatriz reduzida:\n", matriz_reduzida)

#Na linha abaixo, temos o PCA pronto com apenas 2 dimensoes
Y = X_std.dot(matriz_reduzida)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

#Nas linhas 102-111, estamos plotando o gráfico de PCA desse dataset
color = ['navy', 'turquoise', 'darkorange']
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
lw = 2

for color, name in zip(color, targets):
    plt.scatter(Y[y == name, 0], Y[y == name, 1], color=color, alpha=.8, lw=lw,
                label=name)
    
    
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.grid()
plt.show()

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Essa parte do código é dedicada a plotar um biplot

X= X_std 

pca = PCA()
x_new = pca.fit_transform(X) # o pca eh aplicado no dataset 

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

#Call the function. Use only the 2 PCs.
myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
plt.show()
