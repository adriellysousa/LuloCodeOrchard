import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Carregando os dados do arquivo Excel
file_path = r'C:\Users\AdriellyLorenaPalhaS\Documents\TCC2\dataset_preditivo\output\dados_tratados.xlsx'
df = pd.read_excel(file_path)
# Selecionando as colunas para a análise de correlação
colunas_correlacao = [
    'Diametro do dossel', 'Altura da planta', 'Green', 'Red', 'NIR', 'Red_edge', 
    'CLG', 'NDVI', 'Numero de folhas', 'Ramos flores', 'Flores por ramo', 
    'Frutos por ramo', 'Total de frutos'
]

print(colunas_correlacao)
# Calculando a matriz de correlação
matriz_correlacao = df[colunas_correlacao].corr()
# Visualizando a matriz de correlação com um mapa de calor
plt.figure(figsize=(12, 10))
sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Mapa de Calor da Matriz de Correlação')
plt.show()