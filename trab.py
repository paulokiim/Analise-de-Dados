import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

PATH = "iris.data"

names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "Class"]

data = pd.read_csv(PATH, names=names)

# Na linha abaixo nós estamos dando um Drop nas colunas ou linhas vazias do dataset
data.dropna()

# Na linha abaixo nós estamos tirando a coluna "Class" do nosso dataset para que o dataset tenha apenas variáveis numéricas
X = data.drop("Class", 1)

# Na linha abaixo nós estamos guardando em y a coluna "Class" do nosso dataset
y = data["Class"]

# Nesta linha definimos quais são as nossas variáveis de teste e de treinamento para o modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Nas linhas 29-31 estamos normalizando o nosso dataset e ajustando os valores de treinamento de X e de teste de X
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""Nas linhas 38-40, pegamos a matriz de correlação. 
   Fazemos a diagonalização e depois decompus a matriz em seus autovalores e autovetores
   Foi ordenado em ordem decrescente os autovetores com base nos valores absolutos dos seus autovalores
   Foram escolhidos os 2 autovetores que tinham os maiores autovalores 
   Transformei minhas features multiplicando pelos autovetores selecionados
"""
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Na linha abaixo, executamos uma função que explica a variância de cada feature e guardamos em uma variável
explained_variance = pca.explained_variance_ratio_

# Na linha abaixo, printamos a importância que cada feature tem para o dataset e para o que queremos prever
for variance in explained_variance:
    print(variance)

# Na linha abaixo, criamos um classificador e usamos o Random Forest.
classifier = RandomForestClassifier(max_depth=3, random_state=0)
# Na linha abaixo, nós ajustamos os valores de treino de X e os valores de treino de Y para o classificador
classifier.fit(X_train, y_train)

# Na linha abaixo nós realizamos a predição dos valores teste de X e guardamos na variável y_pred
y_pred = classifier.predict(X_test)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
# Na linha abaixo, estamos printando a acurácia da nossa predição após todas os ajustes de PCA
print("Accuracy ", accuracy_score(y_test, y_pred))

