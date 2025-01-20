from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import yaml
import pandas as pd

"""
# Loss combinada: MAE para previsão + penalização da incerteza
def combined_loss(y_true, y_pred):
    y_pred_value, y_pred_uncertainty = y_pred[:, 0], y_pred[:, 1]
    mae_loss = K.mean(K.abs(y_true - y_pred_value))
    uncertainty_penalty = K.mean(y_pred_uncertainty)
    return mae_loss + 0.01 * uncertainty_penalty  # Ajuste do peso do termo de penalização
"""
def mae_loss(y_true, y_pred):
    y_pred_value = y_pred[:, 0] # Apenas a previsão de valor
    mae = K.mean(K.abs(y_true - y_pred_value))  
    return mae


with open(r"C:\Users\ksilva\Documents\Wave_Estimator\config.yml", 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

df = pd.read_csv(config['dataset_path'], sep=',', index_col=None)
df = df.loc[:, ~df.columns.str.contains("Unnamed")]
X = df.drop(['Hs', 'Tp', 'dir'], axis=1)
y = df[config['target']]


X_train = X[:int(0.8 * len(X))]
y_train = y[:int(0.8 * len(y))]
X_test = X[int(0.8 * len(X)):]
y_test = y[int(0.8 * len(y)):]

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler_y.transform(y_test.values.reshape(-1, 1))


input_layer = Input(shape=(165,))
hidden1 = Dense(128, activation='relu')(input_layer)
hidden2 = Dense(64, activation='relu')(hidden1)
hidden3 = Dense(32, activation='relu')(hidden2)

output_prediction = Dense(1, activation='linear', name='prediction')(hidden3)
output_uncertainty = Dense(1, activation='softplus', name='uncertainty')(hidden3)

model = Model(inputs=input_layer, outputs=[output_prediction, output_uncertainty])

#model.compile(optimizer=Adam(learning_rate=0.001),
#              loss={'prediction': 'mae', 'uncertainty': 'mae'},
#              metrics={'prediction': 'mae', 'uncertainty': 'mae'})

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss={'prediction': mae_loss, 'uncertainty': None },  # Apenas o MAE da previsão
    metrics={'prediction': 'mae', 'uncertainty': 'mae'}  # Monitorar incerteza
)

model.summary()

history = model.fit(X_train, {'prediction': y_train, 'uncertainty': y_train}, 
                    epochs=100, batch_size=1, validation_split=0.2)

y_pred, y_uncertainty = model.predict(X_test)

y_test = scaler_y.inverse_transform(y_test)
y_pred = scaler_y.inverse_transform(y_pred)
y_uncertainty = scaler_y.inverse_transform(y_uncertainty)


mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")

print("Predições:", y_pred.flatten())
print("Grau de confiança (incerteza):", y_uncertainty.flatten())


