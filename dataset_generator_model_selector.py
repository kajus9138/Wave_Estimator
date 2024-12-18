import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from scipy.stats import kurtosis, skew
import math
import os

import glob
import warnings

warnings.filterwarnings("ignore")



def gera_features_estatisticas(diretorio):
    arquivos = glob.glob(os.path.join(diretorio, '*'))
    resultados = []

    for idx, arquivo in enumerate(arquivos):
        try:
            # Carregar o arquivo e ignorar as primeiras 4000 linhas
            df = pd.read_csv(arquivo, on_bad_lines='warn', delim_whitespace=True, skiprows=[0])
            df = df[4000:]
            
            
            if df.empty:
                print(f"Arquivo {arquivo} vazio após o corte. Ignorando.")
                continue
            
            # Considerar apenas as 6 primeiras colunas
            df = df.iloc[:, 1:7]
            
            features = {'Arquivo': os.path.basename(arquivo)}
            
            # Calcular as estatísticas para cada coluna
            for coluna in df.columns:
                valores = df[coluna].dropna().to_numpy()  # Ignorar valores NaN
                
                features.update({
                    f'Média_{coluna}': np.mean(valores),
                    f'Mediana_{coluna}': np.median(valores),
                    f'Valor_de_Pico_{coluna}': np.max(valores),
                    f'RMS_{coluna}': np.sqrt(np.mean(valores ** 2)),
                    f'Curtose_{coluna}': kurtosis(valores),
                    f'Assimetria_{coluna}': skew(valores),
                    f'Desvio_Padrão_{coluna}': np.std(valores),
                    f'Amplitude_{coluna}': np.ptp(valores),
                    f'Energia_{coluna}': np.sum(valores ** 2),
                    f'Percentil_25_{coluna}': np.percentile(valores, 25),
                    f'Percentil_75_{coluna}': np.percentile(valores, 75)
                })
            
            resultados.append(features)

        except Exception as e:
            print(f"Erro ao processar o arquivo {arquivo}: {e}")

    return pd.DataFrame(resultados)

diretorio_ondas = r'C:\Users\ksilva\Documents\Wave_Estimator\dados\processados\Ondas_atl_new_fr'
arquivos= glob.glob(os.path.join(diretorio_ondas, '*'))


targets = pd.read_csv(r'C:\Users\ksilva\Documents\Wave_Estimator\dados\processados/Ondas_atl_new_fr.csv', sep=';')

df_features_estatisticas = gera_features_estatisticas(diretorio_ondas)
df_features_estatisticas = pd.concat([df_features_estatisticas,targets['Severidade']], axis=1)
df_features_estatisticas = df_features_estatisticas.T.drop_duplicates().T
df_features_estatisticas.drop('Arquivo', inplace=True, axis=1)

#targets = pd.read_csv(r'C:\Users\ksilva\Documents\Wave_Estimator\dados\processados/Ondas_PMG.csv', sep=';')

#df_features_estatisticas = pd.concat([df_features_estatisticas,targets[['Hs', 'Tp', 'dir']]], axis=1)

df_features_estatisticas.to_csv('dataset_model_selector_atl_new_fr.csv')

#print(df_features_estatisticas)
#print(len(df_features_estatisticas.columns))





