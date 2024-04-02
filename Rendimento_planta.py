from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

# Carregar os dados
dados = pd.read_excel(r'C:\Users\AdriellyLorenaPalhaS\Documents\TCC2\dataset_preditivo\output\dados_tratados.xlsx')

# Definindo a variável alvo e as características
X = dados[['Diametro do dossel', 'Altura da planta', 'Green', 'Red', 'NIR', 'Red_edge', 'CLG', 'NDVI', 'Numero de folhas', 'Ramos flores', 'Flores por ramo', 'Frutos por ramo']]  # Características
y = dados['Total de frutos']  # Variável alvo
# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizando os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construindo e treinando o modelo Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Avaliando o desempenho do modelo no conjunto de treinamento
y_pred_train = model.predict(X_train_scaled)
r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("Desempenho no Conjunto de Treinamento:")
print("R² (Coeficiente de Determinação):", r2_train)
print("RMSE (Raiz do Erro Quadrático Médio):", rmse_train)

# Avaliando o desempenho do modelo no conjunto de teste
y_pred_test = model.predict(X_test_scaled)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("Desempenho no Conjunto de Teste:")
print("R²:", r2_test)
print("RMSE:", rmse_test)
# Preparando o espaço de busca dos hiperparâmetros
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Inicializando o modelo Random Forest
rf = RandomForestRegressor(random_state=42)

# Configurando a busca aleatória com validação cruzada
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Ajustando o modelo ao conjunto de treinamento
random_search.fit(X_train_scaled, y_train)

# Exibindo os melhores hiperparâmetros encontrados
print("Melhores Hiperparâmetros:", random_search.best_params_)

# Selecionando o melhor modelo
best_rf = random_search.best_estimator_

# Seleção de características baseada na importância das características
selector = SelectFromModel(best_rf)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Re-treinando o modelo apenas com as características selecionadas
best_rf.fit(X_train_selected, y_train)

# Avaliação no conjunto de treinamento
y_pred_train_selected = best_rf.predict(X_train_selected)
r2_train_selected = r2_score(y_train, y_pred_train_selected)
rmse_train_selected = np.sqrt(mean_squared_error(y_train, y_pred_train_selected))

print("Desempenho no Conjunto de Treinamento com Características Selecionadas:")
print("R²:", r2_train_selected)
print("RMSE:", rmse_train_selected)

# Avaliação no conjunto de teste
y_pred_test_selected = best_rf.predict(X_test_selected)
r2_test_selected = r2_score(y_test, y_pred_test_selected)
rmse_test_selected = np.sqrt(mean_squared_error(y_test, y_pred_test_selected))

print("Desempenho no Conjunto de Teste com Características Selecionadas:")
print("R²:", r2_test_selected)
print("RMSE:", rmse_test_selected)

# Visualização dos resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(['Treino', 'Teste'], [r2_train, r2_test], color=['blue', 'orange'])
plt.title('Comparação R²')
plt.ylabel('R²')
plt.subplot(1, 2, 2)
plt.bar(['Treino', 'Teste'], [rmse_train, rmse_test], color=['blue', 'orange'])
plt.title('Comparação RMSE')
plt.ylabel('RMSE')
plt.tight_layout()
plt.show()







