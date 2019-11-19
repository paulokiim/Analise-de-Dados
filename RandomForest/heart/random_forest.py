# -*- coding: utf-8 -*-
"""random_forest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10me50PClpchghfGD9wfHDa6Kw2qsPhxs

Começamos importando algumas bibliotecas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

"""Importamos nosso dataset"""

sheets_url = 'https://raw.githubusercontent.com/paulokiim/Analise-de-Dados/master/RandomForest/heart.csv'
df = pd.read_csv(sheets_url)

"""Imprimimos as 10 primeiras linhas"""

print(df.head(10))

"""Calculamos e imprimos quantas pessoas estao saudaveis e quantas estao com doenca de coracao:"""

healthy = df[(df['target'] ==0) ].count()[1]
sick = df[(df['target'] ==1) ].count()[1]

print ("Ha " + str(healthy) + " pessoas saudaveis")
print ("Ha " + str(sick) + " pessoas com problema de coracao")

"""Agora vamos normalizar os dados e separar entre treinamento e testes

O treinamento contemplará 70% do dataset e os testes 30%

Os valores do nosso dataframe eh todo copiado para X. Sao copiados apenas os valores, nao a tabela em si

y contem os valores dos targets: copiados apenas os valores, nao a coluna em si 

iloc seleciona linhas e colunas por números
"""

X = df.iloc[:,0:13].values
y = df.iloc[:,13].values

"""Vamos normalizar o nosso dataset

Criaremos um novo DataFrame, todo normaalizado

Por equanto esse novo data frame nao tem a coluna target
"""

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
dfNorm = pd.DataFrame(X_std, index = df.index, columns = df.columns[0:13])

"""Imprimiremos as 10 primeiras linhas do nosso dataset normalizado, para poder vizualizar"""

print(dfNorm.head(10))

"""Adicionaremos a coluna target ao novo dataframe, normalizado

Após adicionar target, imprimos as 10 primeiras linhas, para vizualizar nossos dados
"""

dfNorm['target'] = df['target']
dfNorm.head(10)

"""Atribuiremos a X os valores do nosso dataframe normalizado

Atribuimos a y os valores do nosso target do nosso dataframe normalizado
"""

X = dfNorm.iloc[:,0:13].values
y = dfNorm.iloc[:,13].values

"""Faremos o treinamento do nosso modelo:"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, y_train.shape, X_test.shape , y_test.shape)

"""Calculamos a matriz de correlação"""

corr = dfNorm.corr()

"""Plotamos o heatmap, ou mapa de calor"""

fig = plt.figure(figsize=(5,4))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.75)

"""Prepara para fazer a função de predição"""

results_test = {}
results_train = {}
list_algos = []

"""Função de predição:"""

def predict_date(algo_name,X_train,y_train,X_test,y_test,atype='',verbose=0):
    algo_name.fit(X_train, y_train)
    Y_pred = algo_name.predict(X_test)
    acc_train = round(algo_name.score(X_train, y_train) * 100, 2)
    acc_val = round(algo_name.score(X_test, y_test) * 100, 2)
    
    results_test[str(algo_name)[0:str(algo_name).find('(')]+'_'+str(atype)] = acc_val
    results_train[str(algo_name)[0:str(algo_name).find('(')]+'_'+str(atype)] = acc_train
    list_algos.append(str(algo_name)[0:str(algo_name).find('(')])
    if verbose ==0:
        print("acc train: " + str(acc_train))
        print("acc test: "+ str(acc_val))
    else:
        return Y_pred

"""Iniciamos o Random Forest com n_estimators:"""

random_forest = RandomForestClassifier(n_estimators=50, random_state = 0)
predict_date(random_forest,X_train,y_train,X_test,y_test)

"""Encontramos a importancia de cada feature:"""

feature_importance = random_forest.feature_importances_
feat_importances = pd.Series(random_forest.feature_importances_, index=df.columns[:-1])
feat_importances = feat_importances.nlargest(13)
feature = df.columns.values.tolist()[0:-1]
importance = sorted(random_forest.feature_importances_.tolist())

"""Estamos plotando o grafico da importancia das features:"""

x_pos = [i for i, _ in enumerate(feature)]
plt.barh(x_pos, importance , color='dodgerblue')
plt.ylabel("feature")
plt.xlabel("importance")
plt.title("feature_importances")
plt.yticks(x_pos, feature)
plt.show()