# Importando bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Carregando o conjunto de dados de preços de imóveis na Califórnia
from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing()

# Convertendo o conjunto de dados para um DataFrame do Pandas
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
data['target'] = california_housing.target

# Dividindo o conjunto de dados em recursos (X) e rótulos (y)
X = data.drop('target', axis=1)
y = data['target']

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliando o desempenho do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualizando as previsões vs. valores reais
plt.scatter(y_test, y_pred)
plt.xlabel("Preços reais")
plt.ylabel("Previsões")
plt.title("Previsões vs. Valores Reais")
plt.show()
