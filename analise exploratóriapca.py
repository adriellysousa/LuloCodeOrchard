from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Carregar os dados
df = pd.read_excel(r'C:\Users\AdriellyLorenaPalhaS\Documents\TCC2\dataset_preditivo\output\dados_tratados.xlsx')

# Remover colunas não numéricas e possíveis identificadores como 'Id'
X = df.select_dtypes(include=[np.number]).drop(['Id'], axis=1)

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA(n_components=5)  
X_pca = pca.fit_transform(X_scaled)

# Variância explicada
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# criar um DataFrame para visualizar a variância explicada por cada componente
pca_df = pd.DataFrame({'Componente': range(1, len(explained_variance) + 1),
                       'Variância explicada': explained_variance})
print(pca_df)

# Obtendo os loadings/vetores próprios da PCA
loadings = pca.components_

# Criando um DataFrame para visualizar os loadings das variáveis
columns = [f'PC{i+1}' for i in range(loadings.shape[0])]
index = X.columns  # as variáveis originais
loadings_df = pd.DataFrame(data=loadings.T, columns=columns, index=index)  # Transpondo os loadings
print(loadings_df)

