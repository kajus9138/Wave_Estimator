import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle
import mlflow
import mlflow.sklearn

df = pd.read_csv(r'C:\Users\ksilva\Documents\Wave_Estimator\dados\processados/dataset_model_selector_atl_new_fr.csv', index_col = 0)
df = shuffle(df, random_state=10)
X = df.drop('Severidade', axis=1)
y = df['Severidade']

mlflow.set_tracking_uri("http://localhost:8000")
experiment_name = "FINEP_model_selector"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)
    model = DecisionTreeClassifier(random_state=10, criterion='gini')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Acurácia: ", acc)

    features_importance = model.feature_importances_
    #mlflow.log_artifact(features_importance)
    mlflow.log_metric('acc',acc)

    mlflow.sklearn.log_model(model, f"decision_tree_model")