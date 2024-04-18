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

#Função biplot
def biplot(score, coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.figure(figsize=(12, 8))
    plt.scatter(xs * scalex, ys * scaley, c='blue', s=5)
    
    # Cria um dicionário para armazenar a posição do texto e evitar sobreposição
    text_positions = {}
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color='red', alpha=0.5)
        if labels is None:
            label = "Var"+str(i+1)
        else:
            label = labels[i]

        # Ajustando a posição do texto para evitar sobreposição
        text_x = coeff[i,0]* 1.15
        text_y = coeff[i,1]* 1.15
        
        # Se 'Red' ou 'Green', ajuste a posição para evitar sobreposição
        if label == 'Red':
            text_x += 0.1
            text_y += 0.02  # Um pouco mais alto
        elif label == 'Green':
            text_x -= 0.1
            text_y -= 0.02  # Um pouco mais baixo
        
        # Armazena a posição do texto para verificar sobreposição
        text_positions[label] = (text_x, text_y)
        
    # Verifica a sobreposição e ajusta as posições
    for label, (x_pos, y_pos) in text_positions.items():
        while any(np.sqrt((x_pos - pos[0])**2 + (y_pos - pos[1])**2) < 0.05 for pos in text_positions.values() if pos != (x_pos, y_pos)):
            x_pos += 0.01
            y_pos += 0.01
        
        plt.text(x_pos, y_pos, label, color='green', ha='center', va='center')
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title('Biplot')
    plt.grid(True)
    plt.show()

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
    # Caminho do arquivo de dados 
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
    
    #Loadings para o biplot
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Chama a função biplot 
    biplot(X_train_pca, loadings, labels=X.columns.tolist())
    
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
# Definindo os nomes dos modelos e as métricas de desempenho
modelos = ['RandomForest', 'XGBoost']
r2_treinamento = [0.92799, 0.90376]
rmse_treinamento = [0.26456, 0.30584]
r2_teste = [0.46328, 0.50638]
rmse_teste = [0.77010, 0.73853]

x = np.arange(len(modelos))  # Posições dos modelos no eixo X
width = 0.35  # Largura das barras

fig, ax = plt.subplots(2, figsize=(10, 8))  # Criando uma figura com dois subplots (um para R² e outro para RMSE)

# R² Score
ax[0].bar(x - width/2, r2_treinamento, width, label='Treinamento', color='skyblue')
ax[0].bar(x + width/2, r2_teste, width, label='Teste', color='orange')
ax[0].set_ylabel('R²')
ax[0].set_title('Comparação do R² por Modelo e Conjunto de Dados')
ax[0].set_xticks(x)
ax[0].set_xticklabels(modelos)
ax[0].legend()

# RMSE
ax[1].bar(x - width/2, rmse_treinamento, width, label='Treinamento', color='skyblue')
ax[1].bar(x + width/2, rmse_teste, width, label='Teste', color='orange')
ax[1].set_ylabel('RMSE')
ax[1].set_title('Comparação do RMSE por Modelo e Conjunto de Dados')
ax[1].set_xticks(x)
ax[1].set_xticklabels(modelos)
ax[1].legend()

fig.tight_layout()  # Ajustando o layout para não haver sobreposição
plt.show()







