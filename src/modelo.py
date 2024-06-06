import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd

# Cargar los datos
data = pd.read_csv('data_consolidated.csv')  # Asegúrate de tener el dataset consolidado
X = data.drop(['reggaeton'], axis=1)
y = data['reggaeton']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Aplicar SMOTE para balancear las clases en los datos de entrenamiento
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Entrenar el modelo
model = GradientBoostingClassifier()
model.fit(X_train_res, y_train_res)

# Evaluar el modelo
accuracy = model.score(X_test, y_test)

# Registrar el modelo en MLFlow
mlflow.set_experiment('Spotify Reggaeton Detection')

with mlflow.start_run():
    mlflow.log_param('model_type', 'Gradient Boosting')
    mlflow.log_metric('accuracy', accuracy)
    
    # Registrar el modelo
    mlflow.sklearn.log_model(model, 'reggaeton_detector')

    # Guardar el modelo en el sistema de archivos
    mlflow.sklearn.save_model(model, 'reggaeton_detector_model')
    
print(f'Model registered with accuracy: {accuracy}')

# Para predecir nuevas instancias
def predict(model, new_data):
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)[:, 1]
    return predictions, probabilities

# Ejemplo de uso
data_test = pd.read_csv('data_test.csv')  # Cargar el dataset de prueba
predictions, probabilities = predict(model, data_test)

# Agregar las predicciones al dataset de prueba
data_test['marca_reggaeton'] = predictions
data_test['probabilidad_reggaeton'] = probabilities

# Guardar las predicciones en un nuevo archivo
data_test.to_csv('data_test_predictions.csv', index=False)
print('Predictions saved to data_test_predictions.csv')
