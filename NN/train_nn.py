from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import yaml
import pandas as pd

# Carregar configuração do arquivo YAML
with open(r"C:\Users\ksilva\Documents\Wave_Estimator\config.yml", 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

# Carregar dados do dataset
df = pd.read_csv(config['dataset_path'], sep=',', index_col=None)
df = df.loc[:, ~df.columns.str.contains("Unnamed")]

# Separar X (features) e y (alvo)
X = df.drop(['Hs', 'Tp', 'dir'], axis=1)
y = df['Hs']

# Dividir em conjuntos de treino e teste (80% treino, 20% teste)
X_train = X[:int(0.8 * len(X))]
y_train = y[:int(0.8 * len(y))]
X_test = X[int(0.8 * len(X)):]
y_test = y[int(0.8 * len(y)):]

# Normalizar os dados de entrada (X)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Normalizar opcionalmente os dados de saída (y)
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

# Definir o modelo
input_layer = Input(shape=(165,))
hidden1 = Dense(128, activation='relu')(input_layer)
hidden2 = Dense(64, activation='relu')(hidden1)
hidden3 = Dense(32, activation='relu')(hidden2)
output = Dense(1, activation='linear', name='output')(hidden3)

model = Model(inputs=input_layer, outputs=output)

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mae',  # Perda de erro absoluto médio
              metrics=['mae'])  # Métrica adicional

model.summary()

# Treinar o modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Fazer predições
y_pred = model.predict(X_test)

# Reverter a normalização das saídas para comparar com os dados reais
y_test = scaler_y.inverse_transform(y_test)
y_pred = scaler_y.inverse_transform(y_pred)

# Avaliar o modelo
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")

print(y_test)
print(y_pred)
