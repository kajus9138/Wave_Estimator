import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score, KFold, ParameterGrid
import yaml

with open(r"C:\Users\ksilva\Documents\Wave_Estimator\Smooth\ATL\config.yml", 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

def calculo_hs(y_test, y_pred):
    limite_superior = y_test * 1.20
    limite_inferior = y_test * 0.80
    fora_da_faixa = np.sum((y_pred > limite_superior) | (y_pred < limite_inferior))
    acc = (len(y_test) - fora_da_faixa) / len(y_test)
    return acc 

def calculo_tp(y_test, y_pred):
    limite_superior = y_test + .5
    limite_inferior = y_test - .5
    fora_da_faixa = np.sum((y_pred > limite_superior) | (y_pred < limite_inferior))
    acc = (len(y_test) - fora_da_faixa) / len(y_test)
    return acc  

def calculo_dir(y_test, y_pred):
    limite_superior = y_test + 10
    limite_inferior = y_test - 10
    fora_da_faixa = np.sum((y_pred > limite_superior) | (y_pred < limite_inferior))
    acc = (len(y_test) - fora_da_faixa) / len(y_test)
    return acc  

def acc_scorer(target, y_test, y_pred):
    calculos = {
        'Hs': calculo_hs,
        'Tp': calculo_tp,
        'dir': calculo_dir,
    }
    return calculos[target](y_test, y_pred)

def train_test_rf(targets):
    df = pd.read_csv(config['dataset_path'], sep=',')
    X = df.drop(['Hs', 'Tp', 'dir', 'Unnamed: 0'], axis=1)

    for target in targets:
        y = df[target]
        param_grid = ParameterGrid(config['param_grid'])

        for params in param_grid:
            mlflow.set_tracking_uri("http://localhost:5000")
            #experiment_name = "FINEP_hd02_RF_cv"
            experiment_name = "FINEP_pj01t_RF_cv"
            #experiment_name = "FINEP_atl_fr_RF_cv"
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run():
                model = RandomForestRegressor(random_state=config['model_params']['random_state'], **params)

                # Configurar validação cruzada
                cv = KFold(n_splits=config['cv_splits'], shuffle=True, random_state=config['random_state'])
                
                # Scorers para MAE, MSE e acc
                mae_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
                mse_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
                rmse_scores = np.sqrt(-mse_scores)

                # Cálculo da acurácia usando o scorer personalizado
                acc_scores = []
                feature_importances_list = []  # Para armazenar as importâncias de todas as dobras
                
                for train_idx, test_idx in cv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc_scores.append(acc_scorer(target, y_test, y_pred))
                    
                    # Adiciona as feature importances da dobra atual
                    feature_importances_list.append(model.feature_importances_)

                # Média das métricas de validação cruzada
                mae_avg = -np.mean(mae_scores)
                mse_avg = -np.mean(mse_scores)
                rmse_avg = np.mean(rmse_scores)
                acc_avg = np.mean(acc_scores)

                # Calcular a média das feature importances
                mean_importances = np.mean(feature_importances_list, axis=0)
                feature_importances_df = pd.DataFrame({
                    "feature": X.columns,
                    "importance": mean_importances
                })
                
                # Salvar como artefato no MLflow
                feature_importances_file = f"feature_importances_{target}.csv"
                feature_importances_df.to_csv(feature_importances_file, index=False)
                mlflow.log_artifact(feature_importances_file)

                # Logar parâmetros e métricas no MLflow
                mlflow.log_params(params)
                mlflow.log_param("target", target)
                mlflow.log_param("dataset_path", config['dataset_path'])
                mlflow.log_param("cv_splits", config['cv_splits'])
                mlflow.log_param("random_state", config['random_state'])

                mlflow.log_metric("mae_avg", mae_avg)
                mlflow.log_metric("mse_avg", mse_avg)
                mlflow.log_metric("rmse_avg", rmse_avg)
                mlflow.log_metric("acc_avg", acc_avg)

                # Logar o modelo treinado
                mlflow.sklearn.log_model(model, f"random_forest_model_{target}")

                # Exibir as métricas no console para verificar
                print(f"\nTarget: {target}")
                print("Parâmetros:", params)
                print("Métricas de validação cruzada:")
                print(f"MAE (Validação Cruzada): {mae_avg}")
                print(f"MSE (Validação Cruzada): {mse_avg}")
                print(f"RMSE (Validação Cruzada): {rmse_avg}")
                print(f"Acurácia (Validação Cruzada): {acc_avg}")
                print(f"Feature Importances salvas em: {feature_importances_file}")

# Executar a função com a lista de targets
train_test_rf(targets=config['target'])
