import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from preprocess import load_and_preprocess_data

# Carregar os dados pré-processados
file_path = "data\\agaricus-lepiota.data"
df, _ = load_and_preprocess_data(file_path)

# Definir as features reduzidas
selected_features = ["odor", "spore-print-color", "stalk-surface-below-ring", 
                     "stalk-color-above-ring", "habitat", "cap-color"]

X_reduced = df[selected_features]
y = df["class"]

# Dividir os dados em treino e teste (70% treino, 30% teste para otimizar o tempo)
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_reduced, y, test_size=0.3, random_state=42, stratify=y)

# Definir os hiperparâmetros fixos
best_params = {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}

# Criar e treinar o modelo reduzido com os hiperparâmetros fixos
print(f"Treinando o modelo reduzido SVM com hiperparâmetros fixos: {best_params}")
model_red = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
model_red.fit(X_train_red, y_train_red)

# Criar diretório para salvar modelo otimizado
model_dir = "src"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Salvar o modelo treinado
joblib.dump(model_red, os.path.join(model_dir, "svm_model_fixed_reduced02.pkl"))
print("Modelo reduzido treinado e salvo com sucesso!")
