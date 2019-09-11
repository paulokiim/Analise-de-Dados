import pandas as pd
import numpy as np
import plotly.graph_objects as go
import chart_studio.plotly as py
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

# Na linha abaixo, estamos fazendo uma lista de tuplas com (autovalor, autovetor)
pares = [
    (np.abs(autovalores_correlacao[i]), autovetores_correlacao[:, i])
    for i in range(len(autovalores_correlacao))
]

# Nas linhas 66 e 67, estamos ordenando nossas tuplas decrescentemente
pares.sort()
pares.reverse()

# Nas linhas 70-72, estamos confirmando que nossa lista foi ordenada corretamente
print("Autovalores em ordem decrescente")
for i in pares:
    print(i[0])

# Nas linhas 76-78, estamos criando a nossa explicacao da variancia
tot = sum(autovalores_correlacao)
explicacao_variancia = [
    (i / tot) * 100 for i in sorted(autovalores_correlacao, reverse=True)
]
variancia_cum = np.cumsum(explicacao_variancia )
# Na linha abaixo, estamos printando a explicacao da variancia de cada feature
print("\nExplicacao das variancias\n",explicacao_variancia)

#Na linha abaixo estamos reduzindo a nossa matriz para dimensao 2
matriz_reduzida = np.hstack((pares[0][1].reshape(4, 1), pares[1][1].reshape(4, 1)))

#Na linha abaixo estamos printando a nossa matriz reduzida
print("\nMatriz reduzida:\n", matriz_reduzida)

Y = X_std.dot(matriz_reduzida)

total_rows = data.count
number_of_iris = len(data)




data["ID"] = data.index
data["ratio"] = data["sepal-length"]/data["sepal-width"]
sns.lmplot(x="ID", y="ratio", data=data, hue="Class", fit_reg=False, legend=False)

plt.legend()
plt.show()

'''

trace1 = dict(
    type='bar',
    x=['PC %s' %i for i in range(1,5)],
    y=explicacao_variancia,
    name='Individual'
)

trace2 = dict(
    type='scatter',
    x=['PC %s' %i for i in range(1,5)], 
    y= variancia_cum,
    name='Cumulative'
)

dados = [trace1, trace2]

layout=dict(
    title='Explained variance by different principal components',
    yaxis=dict(
        title='Explained variance in percent'
    ),
    annotations=list([
        dict(
            x=1.16,
            y=1.05,
            xref='paper',
            yref='paper',
            text='Explained Variance',
            showarrow=False,
        )
    ])
)

fig = dict(data=dados, layout=layout)
py.iplot(fig, filename='selecting-principal-components')

# plotting histograms
df = []

legend = {0: False, 1: False, 2: False, 3: True}

colors = {
    "Iris-setosa": "#0D76BF",
    "Iris-versicolor": "#00cc96",
    "Iris-virginica": "#EF553B",
}

for col in range(4):
    for key in colors:
        trace = dict(
            type="histogram",
            x=list(X[y == key, col]),
            opacity=0.75,
            xaxis="x%s" % (col + 1),
            marker=dict(color=colors[key]),
            name=key,
            showlegend=legend[col],
        )
        data.append(trace)

layout = dict(
    barmode="overlay",
    xaxis=dict(domain=[0, 0.25], title="sepal length (cm)"),
    xaxis2=dict(domain=[0.3, 0.5], title="sepal width (cm)"),
    xaxis3=dict(domain=[0.55, 0.75], title="petal length (cm)"),
    xaxis4=dict(domain=[0.8, 1], title="petal width (cm)"),
    yaxis=dict(title="count"),
    title="Distribution of the different Iris flower features",
)

fig = dict(data=df, layout=layout)
py.iplot(fig, filename="exploratory-vis-histogram")


'''