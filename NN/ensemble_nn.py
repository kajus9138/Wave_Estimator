from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import yaml
import numpy as np
import pandas as pd

# Função de perda MAE
def mae_loss(y_true, y_pred):
    y_pred_value = y_pred[:, 0]  # Apenas a previsão de valor
    mae = K.mean(K.abs(y_true - y_pred_value))
    return mae


# Configuração de caminhos e leitura de dados
with open(r"C:\Users\ksilva\Documents\Wave_Estimator\config.yml", 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

df = pd.read_csv(config['dataset_path'], sep=',', index_col=None)
df = df.loc[:, ~df.columns.str.contains("Unnamed")]
X = df.drop(['Hs', 'Tp', 'dir'], axis=1)
y = df[config['target']]

# Divisão dos dados
X_train = X[:int(0.8 * len(X))]
y_train = y[:int(0.8 * len(y))]
X_test = X[int(0.8 * len(X)):]
y_test = y[int(0.8 * len(y)):]

# Normalização dos dados
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler_y.transform(y_test.values.reshape(-1, 1))


# Função para criar o modelo
def create_model():
    input_layer = Input(shape=(X_train.shape[1],))
    hidden1 = Dense(128, activation='relu')(input_layer)
    hidden2 = Dense(64, activation='relu')(hidden1)
    hidden3 = Dense(32, activation='relu')(hidden2)
    output_prediction = Dense(1, activation='linear', name='prediction')(hidden3)

    model = Model(inputs=input_layer, outputs=output_prediction)
    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss='mae',
                  metrics=['mae'])
    return model


# Treinando um ensemble de modelos
n_models = 20  # Número de modelos no ensemble
ensemble_models = []

for i in range(n_models):
    print(f"Treinando modelo {i+1}/{n_models}")
    model = create_model()
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    ensemble_models.append(model)

# Fazendo previsões com o ensemble
predictions = np.array([model.predict(X_test) for model in ensemble_models])
mean_prediction = np.mean(predictions, axis=0)  # Média das previsões
variance_prediction = np.var(predictions, axis=0)  # Variância das previsões

# Revertendo normalização
y_test = scaler_y.inverse_transform(y_test)
mean_prediction = scaler_y.inverse_transform(mean_prediction)
variance_prediction = variance_prediction.flatten()  # Não precisa de inversão

# Calculando o erro
mae = mean_absolute_error(y_test, mean_prediction)
print(f"MAE: {mae}")

# Resultados
print("Predições médias:", mean_prediction.flatten())
print("Incerteza (variância):", variance_prediction)
