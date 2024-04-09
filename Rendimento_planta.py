# Importando bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Função para carregar os dados
def carregar_dados(caminho):
    return pd.read_excel(caminho)

# Função para tratamento de valores ausentes
def tratar_valores_ausentes(dados, variaveis_de_interesse):
    return dados[variaveis_de_interesse].fillna(dados[variaveis_de_interesse].mean())

# Função para visualização da distribuição da variável alvo e mapa de calor de correlações
def visualizar_dados(dados):
    plt.figure(figsize=(12, 6))
    sns.histplot(dados['Total de frutos'], kde=True, bins=30, color="skyblue")
    plt.title('Distribuição do Total de Frutos')
    plt.xlabel('Total de Frutos')
    plt.ylabel('Frequência')
    plt.show()

    plt.figure(figsize=(12, 10))
    sns.heatmap(dados.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Mapa de Calor da Matriz de Correlação')
    plt.show()

# Função para padronização dos dados
def padronizar_dados(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Função para aplicar PCA
def aplicar_pca(X_scaled, n_components=5):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca

# Função para ajuste fino e treinamento de modelos
def ajuste_fino_treinamento(X_train, X_test, y_train, y_test):
    modelos = {
        'RandomForestRegressor': {
            'modelo': RandomForestRegressor(random_state=42),
            'parametros': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
            }
        },
        'XGBRegressor': {
            'modelo': XGBRegressor(random_state=42),
            'parametros': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1],
            }
        }
    }

    resultados = {}
    for nome, d in modelos.items():
        grid_search = GridSearchCV(estimator=d['modelo'], param_grid=d['parametros'], cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        melhor_modelo = grid_search.best_estimator_
        y_pred_train = melhor_modelo.predict(X_train)
        y_pred_test = melhor_modelo.predict(X_test)
        resultados[nome] = {
            'Melhores Parâmetros': grid_search.best_params_,
            'R2 Treinamento': r2_score(y_train, y_pred_train),
            'RMSE Treinamento': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'R2 Teste': r2_score(y_test, y_pred_test),
            'RMSE Teste': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        }
    return resultados

# Função principal
def main():
    # Caminho do arquivo de dados (ajuste conforme necessário)
    caminho_dados = r'C:\Users\AdriellyLorenaPalhaS\Documents\TCC2\dataset_preditivo\output\dados_tratados.xlsx'
    dados = carregar_dados(caminho_dados)
    
    # Variáveis de interesse
    variaveis_de_interesse = [
        'Diametro do dossel', 'Altura da planta', 'Green', 'Red', 'NIR', 'Red_edge',
        'NDVI', 'Numero de folhas', 'Ramos flores', 'Flores por ramo', 'Frutos por ramo', 'Total de frutos'
    ]
    dados_selecionados = tratar_valores_ausentes(dados, variaveis_de_interesse)

    # Visualizações de dados
    visualizar_dados(dados_selecionados)

    # Divisão dos dados em conjuntos de treino e teste
    X = dados_selecionados.drop(['Total de frutos'], axis=1)
    y = dados_selecionados['Total de frutos']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Padronização dos dados
    X_train_scaled, scaler = padronizar_dados(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Aplicar PCA
    X_train_pca, pca = aplicar_pca(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Ajuste fino e treinamento dos modelos
    resultados = ajuste_fino_treinamento(X_train_pca, X_test_pca, y_train, y_test)
    for modelo, metricas in resultados.items():
        print(f"{modelo}:")
        print(f"Melhores Parâmetros: {metricas['Melhores Parâmetros']}")
        print(f"R2 Treinamento: {metricas['R2 Treinamento']}, RMSE Treinamento: {metricas['RMSE Treinamento']}")
        print(f"R2 Teste: {metricas['R2 Teste']}, RMSE Teste: {metricas['RMSE Teste']}")
        print('---')

if __name__ == '__main__':
    main()








