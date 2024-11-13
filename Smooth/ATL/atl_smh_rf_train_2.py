import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
import yaml
from sklearn.model_selection import ParameterGrid

with open("config.yml", 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

def train_test_rf(target):

    df = pd.read_csv(config['dataset_path'], sep=',')
    X = df.drop(['Hs', 'Tp', 'dir'], axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=config['random_state'])

    param_grid = config['param_grid']
    model = RandomForestRegressor(random_state=config['model_params']['random_state'])
    param_grid = ParameterGrid(config['param_grid'])  # Gera combinações de parâmetros

    for params in param_grid:

        mlflow.set_tracking_uri("http://localhost:5000")
        experiment_name = "FINEP_Espectral_Features_RF"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            
            model = RandomForestRegressor(random_state=config['model_params']['random_state'], **params)
            model.fit(X_train, y_train)
            
            # Previsões e métricas de treino
            y_train_pred = model.predict(X_train)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            mse_train = mean_squared_error(y_train, y_train_pred)
            rmse_train = np.sqrt(mse_train)

            # Previsões e métricas de teste
            y_test_pred = model.predict(X_test)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)
            rmse_test = np.sqrt(mse_test)
            
            # Logar parâmetros e métricas
            mlflow.log_params(params)
            mlflow.log_param("dataset_path", config['dataset_path'])
            mlflow.log_param("test_size", config['test_size'])
            mlflow.log_param("random_state", config['random_state'])

            mlflow.log_metric("mae_train", mae_train)
            mlflow.log_metric("mse_train", mse_train)
            mlflow.log_metric("rmse_train", rmse_train)
            mlflow.log_metric("mae_test", mae_test)
            mlflow.log_metric("mse_test", mse_test)
            mlflow.log_metric("rmse_test", rmse_test)

            # Logar o modelo treinado
            mlflow.sklearn.log_model(model, "random_forest_model")

            # Exibir as métricas no console para verificar
            print("Parâmetros:", params)
            print("Métricas de treino:")
            print(f"MAE (Treinamento): {mae_train}")
            print(f"MSE (Treinamento): {mse_train}")
            print(f"RMSE (Treinamento): {rmse_train}")

            print("\nMétricas de teste:")
            print(f"MAE (Teste): {mae_test}")
            print(f"MSE (Teste): {mse_test}")
            print(f"RMSE (Teste): {rmse_test}")

train_test_rf(target=config['target'])