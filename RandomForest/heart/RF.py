# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 00:29:39 2019
@author: kuo incrível, zap
"""

# imports 
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

#matplotlib inline
df = pd.read_csv('heart.csv')

#print first 10 rows
print(df.head(10))

healthy = df[(df['target'] ==0) ].count()[1]
#Calcula quantas pessoas estao saudaveis

sick = df[(df['target'] ==1) ].count()[1]
#Calcula quantas pessoas estao com problema de coracao

print ("Ha " + str(healthy) + " pessoas saudaveis")
print ("Ha " + str(sick) + " pessoas com problema de coracao")

#Agora vamos normalizar os dados e separar entre treinamento e testes
#O treinamento contemplará 70% do dataset e os testes 30%

X = df.iloc[:,0:13].values
#Os valores do nosso dataframe eh todo copiado para X
#Sao copiados apenas os valores, nao a tabela em si

y = df.iloc[:,13].values
# y contem os valores dos targets
# Copiados apenas os valores, nao a coluna em si 

'''iloc seleciona linhas e colunas por números'''

#Vamos normalizar o nosso dataset
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
dfNorm = pd.DataFrame(X_std, index = df.index, columns = df.columns[0:13])
#Criamos um novo DataFrame, todo normaalizado
#Por equanto esse novo data frame nao tem a coluna target

print()

print(dfNorm.head(10))
#Imprimimos as 10 primeiras linhas dos nossos dados

print()

#Adicionaremos a coluna target ao novo dataframe, normalizado
dfNorm['target'] = df['target']

#Imprimos as 10 primeiras linhas, para vizualizar nossos dados
dfNorm.head(10)

#Atribuiremos a X os valores do nosso dataframe normalizado
X = dfNorm.iloc[:,0:13].values

print()
print(X)

#Atribuimos a y os valores do nosso target do nosso dataframe normalizado
y = dfNorm.iloc[:,13].values

#Faremos o treinamento do nosso modelo
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=0)
X_train.shape, y_train.shape, X_test.shape , y_test.shape

# Aqui calculamos a matriz de correlacao
corr = dfNorm.corr()

# Plotamos o heatmap, ou mapa de calor
fig = plt.figure(figsize=(5,4))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
            linewidths=.75)


results_test = {}
results_train = {}
list_algos = []

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

# Iniciamos o Random Forest com n_estimators 
random_forest = RandomForestClassifier(n_estimators=50, random_state = 0)
predict_date(random_forest,X_train,y_train,X_test,y_test)

# Encontramos a importancia de cada feature
feature_importance = random_forest.feature_importances_
feat_importances = pd.Series(random_forest.feature_importances_, index=df.columns[:-1])
feat_importances = feat_importances.nlargest(13)
feature = df.columns.values.tolist()[0:-1]
importance = sorted(random_forest.feature_importances_.tolist())

# Estamos plotando o grafico da importancia das features
x_pos = [i for i, _ in enumerate(feature)]

plt.barh(x_pos, importance , color='dodgerblue')
plt.ylabel("feature")
plt.xlabel("importance")
plt.title("feature_importances")

plt.yticks(x_pos, feature)

plt.show()
