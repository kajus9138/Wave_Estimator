import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
from scipy.special import gamma  # Função Gamma
from scipy.stats import kurtosis, skew
import math
import os
from scipy.signal import csd
from scipy.interpolate import interp1d
import glob
import warnings

warnings.filterwarnings("ignore")

def A(s):
   
    return (2**(2*s-1)) * (sp.gamma(s + 1)**2) / (np.pi * sp.gamma(2*s + 1))

def E_g(omega, theta, lambda_vals, Hs_vals, omega_m_vals, theta_m_vals, s_vals):

    E_g_sum = 0
    
    for i in range(2):
        lambda_i = lambda_vals[i]
        Hs_i = Hs_vals[i]
        omega_m_i = omega_m_vals[i]
        theta_m_i = theta_m_vals[i]
        s_i = s_vals[i]
        
        Gamma_lambda_i = sp.gamma(lambda_i)
        
        A_si = A(s_i)
        
        term = ((4*lambda_i + 1)/4) * omega**(4 * lambda_i) * (lambda_i / Gamma_lambda_i)
        term *= Hs_i**2 * omega**(4*lambda_i + 1) * A_si
        term *= np.cos((theta - theta_m_i) / 2)**(2*s_i)
        term *= np.exp(-(4*lambda_i + 1)/4 * (omega_m_i / omega)**4)
        
        E_g_sum += term
    
    E_g_final = E_g_sum / 4
    
    return E_g_final

def compute_cpsd(signal1, signal2, fs, freqs_rad):
    f, Pxy = csd(signal1, signal2, fs=fs, nperseg=1800)
    f_rad = 2 * np.pi * f  
    
    interp_func_real = interp1d(f_rad, np.real(Pxy), kind='linear', fill_value='extrapolate')
    interp_func_imag = interp1d(f_rad, np.imag(Pxy), kind='linear', fill_value='extrapolate')
    
    Pxy_real_interp = interp_func_real(freqs_rad)
    Pxy_imag_interp = interp_func_imag(freqs_rad)
    
    return Pxy_real_interp, Pxy_imag_interp


def plot_cpsd(signal1, signal2, label1, label2, fs):
    f, Pxy = csd(signal1, signal2, fs=fs, nperseg=1800)
    plt.plot(f, np.real(Pxy), 'b', label=f'Real ({label1} vs {label2})')
    plt.plot(f, np.imag(Pxy), 'r', label=f'Imaginary ({label1} vs {label2})')


def gera_espectros(diretorio):
    arquivos = glob.glob(diretorio + '/*')
    
    fs = 1.0 
    frequencies_rad = np.linspace(0.2,3,100)
    combined_matrices = np.zeros((len(arquivos), 9, 100))
    
    
    for idx, arquivo in enumerate(arquivos):
        
        
        df = pd.read_csv(arquivo, sep=';') 
      
        
        
        #sway = df['Y.1'].values
        sway = df['Y'].values
        heave = df['Z'].values
        pitch = df['Y-Y'].values
        
        signal_pairs = [
            (sway, heave),   # 1
            (sway, pitch),   # 2
            (heave, pitch),  # 3
            (sway, sway),    # 4 (autoespectro)
            (heave, heave),  # 5 (autoespectro)
            (pitch, pitch)   # 6 (autoespectro)
        ]
        
    
        matrix_real = np.zeros((6, 100))
        matrix_imag = np.zeros((3, 100))
        
        
        for i, (sig1, sig2) in enumerate(signal_pairs):
            real_part, imag_part = compute_cpsd(sig1, sig2, fs, frequencies_rad)
            if i < 3:  # Para as 3 primeiras combinações, guardar parte real e imaginária
                matrix_real[i] = real_part
                matrix_imag[i] = imag_part
            else:  # Para autoespectros, apenas a parte real
                matrix_real[i] = real_part  
        
        
        combined_matrix = np.vstack([matrix_real, matrix_imag])    
        combined_matrices[idx] = combined_matrix

        print(combined_matrices.shape)

    return combined_matrices


def gera_features_estatisticas_espectrais(signal):
    resultados = []
    
    for onda in range(signal.shape[0]):
        features_onda = {"Onda": onda}
        
        for linha in range(signal.shape[1]):
            espectro = signal[onda, linha, :]
            
            # Calculando as features estatísticas
            features_onda[f"Média_{linha+1}"] = np.mean(espectro)
            features_onda[f"Mediana_{linha+1}"] = np.median(espectro)
            features_onda[f"ValorPico_{linha+1}"] = np.max(espectro)
            features_onda[f"RMS_{linha+1}"] = np.sqrt(np.mean(espectro**2))
            features_onda[f"Kurtosis_{linha+1}"] = kurtosis(espectro)
            features_onda[f"Skewness_{linha+1}"] = skew(espectro)
            features_onda[f"DesvioPadrão_{linha+1}"] = np.std(espectro)
            features_onda[f"Amplitude_{linha+1}"] = np.max(espectro) - np.min(espectro)
            features_onda[f"Energia_{linha+1}"] = np.sum(espectro**2)
            features_onda[f"Percentil25_{linha+1}"] = np.percentile(espectro, 25)
            features_onda[f"Percentil75_{linha+1}"] = np.percentile(espectro, 75)
        
        resultados.append(features_onda)
    
    df_features = pd.DataFrame(resultados)
    df_features.set_index('Onda', inplace=True)

    print(df_features.head(1))
    
    return df_features



def gera_features_estatisticas(diretorio):
    arquivos = glob.glob(os.path.join(diretorio, '*'))
    resultados = []

    for idx, arquivo in enumerate(arquivos):
        try:
            # Carregar o arquivo e ignorar as primeiras 4000 linhas
           # df = pd.read_csv(arquivo, on_bad_lines='warn', delim_whitespace=True, skiprows=[0])
            df = pd.read_csv(arquivo, sep=';')
            df = df.loc[:, ~df.columns.str.contains("Unnamed")] 
            print("criou df")
            print(df)
            #df = df[4000:]
            
            
            if df.empty:
                print(f"Arquivo {arquivo} vazio após o corte. Ignorando.")
                continue
            
            # Considerar apenas as 6 primeiras colunas
            #df = df.iloc[:, 0:7]
            #df = df.iloc[:, 1:7] usado para os arquivos txt simulados
            
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

diretorio_ondas = r'C:\Users\ksilva\Documents\Wave_Estimator\dados\processados\Ondas_reais_ago2024'
arquivos= glob.glob(os.path.join(diretorio_ondas, '*'))

combinacoes_espectrais = gera_espectros(diretorio_ondas)
features_espectrais = gera_features_estatisticas_espectrais(combinacoes_espectrais)
df_features_estatisticas = gera_features_estatisticas(diretorio_ondas)

print(df_features_estatisticas.head(1))

df_features_estatisticas = pd.concat([features_espectrais, df_features_estatisticas], axis=1)
#df_features_estatisticas = df_features_estatisticas.T.drop_duplicates().T
df_features_estatisticas.drop('Arquivo', inplace=True, axis=1)

#targets = pd.read_csv(r'C:\Users\ksilva\Documents\Wave_Estimator\dados\processados/Ondas_PMG.csv', sep=';')

#df_features_estatisticas = pd.concat([df_features_estatisticas,targets[['Hs', 'Tp', 'dir']]], axis=1)

df_features_estatisticas.to_csv('dataset_02_atl_fr_val4_26_27real.csv')

#print(df_features_estatisticas)
#print(len(df_features_estatisticas.columns))





