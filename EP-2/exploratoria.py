import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn import svm
from sklearn.model_selection import train_test_split

PATH = 'bfi.csv'

data = pd.read_csv(PATH)

# Dropando colunas desnecessarias
data.drop(['Unnamed: 0', 'gender', 'education', 'age'], axis=1, inplace=True)

# Dropando colunas sem valores
data.dropna(inplace=True)

# Teste da Esferacidade de Bartlett!!
chi_square_value, p_value = calculate_bartlett_sphericity(data)
print(chi_square_value, p_value)
# p_value = 0, entao podemos proceder com a analise de fatores pois quanto menor esse valor, mais confianca temos nas predicoes

# Teste de Kiaser-Meyer-Olkin(KMO)
kmo_all, kmo_model = calculate_kmo(data)
print(kmo_model)
# Resultado 0.84, entao podemos proceder com a nossa analise de fatores

# Criamos um objeto analise de fatores sem rotacao
analisador_sem_rotacao = FactorAnalyzer(n_factors=20, rotation=None)
analisador_sem_rotacao.fit(data)
# Aqui estamos checando os nossos autovalores
autovalores, v = analisador_sem_rotacao.get_eigenvalues()
print(autovalores)

# Criamos o Grafico Scree para observar quais autovalores sao maiores que 1, neste caso usaremos 6 fatores
plt.scatter(range(1, data.shape[1]+1), autovalores)
plt.plot(range(1, data.shape[1]+1), autovalores)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

# Criamos um objeto analise de faotres com rotacao varimax
analisador_varimax = FactorAnalyzer(n_factors=5, rotation="varimax")
analisador_varimax.fit(data)

autovalores_varimax, v = analisador_varimax.get_eigenvalues()
print (autovalores_varimax)
# Nesta linha conseguimos ver que a variancia cumulativa chega a 42% com 5 fatores
print(analisador_varimax.get_factor_variance())

# Criamos um objeto analise de faotres com rotacao quartimax
analisador_quartimax = FactorAnalyzer(n_factors=5, rotation="quartimax")
analisador_quartimax.fit(data)

autovalores_quartimax, v = analisador_quartimax.get_eigenvalues()
print (autovalores_quartimax)
# Nesta linha conseguimos ver que a variancia cumulativa chega a 42% com 5 fatores
print(analisador_quartimax.get_factor_variance())

# Criamos um objeto analise de faotres com rotacao promax
analisador_promax = FactorAnalyzer(n_factors=5, rotation="promax")
analisador_promax.fit(data)

autovalores_promax, v = analisador_promax.get_eigenvalues()
print (autovalores_promax)
# Nesta linha conseguimos ver que a variancia cumulativa chega a 41% com 5 fatores
print(analisador_promax.get_factor_variance())

#Podemos observar que a variancia cumulativa para Quartimax e Varimax(Ortogonal) sao as mesmas porem para Promax(Obliquo) é diferente!

X_train, X_test, y_train, y_test = train_test_split(data, data.columns, test_size=0.4, random_state=0)

print (X_train.shape, y_train.shape)