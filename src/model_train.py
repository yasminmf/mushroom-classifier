import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from preprocess import load_and_preprocess_data

# Carregar os dados pré-processados
file_path = "C:/Users/yasmi/INTELIGENCIA ARTIFICIAL/mushroom-classifier/data/agaricus-lepiota.data"
df, _ = load_and_preprocess_data(file_path)

# Definir as features reduzidas
selected_features = ["odor", "spore-print-color", "stalk-surface-below-ring", 
                     "stalk-color-above-ring", "habitat", "cap-color"]

X_reduced = df[selected_features]
y = df["class"]

# Dividir os dados em treino e teste (70% treino, 30% teste para otimizar o tempo)
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_reduced, y, test_size=0.3, random_state=42, stratify=y)

# Definir os hiperparâmetros reduzidos para otimizar tempo
param_grid = {
    'kernel': ['rbf'],  # Testando apenas o kernel RBF
    'C': [1, 10],  # Reduzindo para 2 valores
    'gamma': ['scale']  # Fixando gamma
}

# Criar e otimizar modelo reduzido com GridSearchCV
print("Buscando melhores hiperparâmetros para o modelo reduzido...")
model_red = SVC()
grid_search_red = GridSearchCV(model_red, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_red.fit(X_train_red, y_train_red)
best_model_red = grid_search_red.best_estimator_
best_params_red = grid_search_red.best_params_
print(f"Melhores hiperparâmetros para o modelo reduzido: {best_params_red}")
print(f"Acurácia do melhor modelo reduzido: {grid_search_red.best_score_ * 100:.2f}%")

# Criar diretório para salvar modelo otimizado
model_dir = "../model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Salvar o melhor modelo treinado
joblib.dump(best_model_red, os.path.join(model_dir, "svm_model_gridsearch_reduced.pkl"))
print("Modelo reduzido otimizado treinado e salvo com sucesso!")
