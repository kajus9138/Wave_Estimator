## Import de libs

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.special as sp
from scipy.special import gamma  # Função Gamma
from scipy.stats import kurtosis, skew
import math
import os
from scipy.signal import csd
from scipy.interpolate import interp1d
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import mlflow
import yaml

with open("config.yml", 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

#df = pd.read_csv(r'C:\Users\ksilva\Documents\finep dados\IA_ML\ML\Smooth\ATL/dataset_smh_01.csv', sep=',')

df = pd.read_csv(config['dataset_path'], sep=',')
X = df.drop(['Hs', 'Tp', 'dir'], axis=1)
y = df['Hs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=config['random_state'])

with mlflow.start_run():
    # Logar parâmetros
    mlflow.log_param("dataset_path", config['dataset_path'])
    mlflow.log_param("test_size", config['test_size'])
    mlflow.log_param("random_state", config['random_state'])
    mlflow.log_param("n_estimators", config['model_params']['n_estimators'])
    mlflow.log_param("model_random_state", config['model_params']['random_state'])
    mlflow.log_param("criterion", config['model_params']['criterion'])
    mlflow.log_param("max_depth", config['model_params']['max_depth'])
    mlflow.log_param("min_samples_split", config['model_params']['min_samples_split'])
    mlflow.log_param("min_samples_leaf", config['model_params']['min_samples_leaf'] )
    

    # Criar e treinar o modelo RandomForestRegressor
    model = RandomForestRegressor(n_estimators=config['model_params']['n_estimators'], 
                                  random_state=config['model_params']['random_state'],
                                  criterion=config['model_params']['criterion'],
                                  max_depth=config['model_params']['max_depth'],
                                  min_samples_split=config['model_params']['min_samples_split'],
                                  min_samples_leaf=config['model_params']['min_samples_leaf'])
    model.fit(X_train, y_train)

    # Fazer previsões no conjunto de treino
    y_train_pred = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)

    # Fazer previsões no conjunto de teste
    y_test_pred = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)

    # Logar métricas
    mlflow.log_metric("mae_train", mae_train)
    mlflow.log_metric("mse_train", mse_train)
    mlflow.log_metric("rmse_train", rmse_train)

    mlflow.log_metric("mae_test", mae_test)
    mlflow.log_metric("mse_test", mse_test)
    mlflow.log_metric("rmse_test", rmse_test)

    # Logar o modelo treinado
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Exibir as métricas no console
    print("Métricas de treino:")
    print(f"MAE (Treinamento): {mae_train}")
    print(f"MSE (Treinamento): {mse_train}")
    print(f"RMSE (Treinamento): {rmse_train}")

    print("\nMétricas de teste:")
    print(f"MAE (Teste): {mae_test}")
    print(f"MSE (Teste): {mse_test}")
    print(f"RMSE (Teste): {rmse_test}")





'''
model = RandomForestRegressor(n_estimators=config['model_params']['n_estimators'],
                               random_state=config['model_params']['random_state'])
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)

y_test_pred = model.predict(X_test)
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)



print("Métricas de treino:")
print(f"MAE (Treinamento): {mae_train}")
print(f"MSE (Treinamento): {mse_train}")
print(f"RMSE (Treinamento): {rmse_train}")

print("\nMétricas de teste:")
print(f"MAE (Teste): {mae_test}")
print(f"MSE (Teste): {mse_test}")
print(f"RMSE (Teste): {rmse_test}")

'''
