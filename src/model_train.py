import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from preprocess import load_and_preprocess_data
from sklearn.model_selection import RandomizedSearchCV

# Carregar os dados pré-processados
file_path = "C:\\Users\\Arlison Gaspar\\Desktop\\testIA\\mushroom-classifier\\data\\agaricus-lepiota.data"
df, _ = load_and_preprocess_data(file_path)

# Definir as features reduzidas
selected_features = ["odor", "spore-print-color", "stalk-surface-below-ring", 
                     "stalk-color-above-ring", "habitat", "cap-color"]

X_reduced = df[selected_features]
y = df["class"]

# Dividir os dados em treino e teste (70% treino, 30% teste para otimizar o tempo)
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_reduced, y, test_size=0.3, random_state=42, stratify=y)

# Definir o espaço de busca de hiperparâmetros para o Grid Search
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # 5 valores
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # 5 valores
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']  # 4 valores
}

# Configurar o Grid Search
random_search = RandomizedSearchCV(
    estimator=SVC(),
    param_distributions=param_grid,
    n_iter=90,  # Número de combinações aleatórias a serem testadas
    scoring='accuracy',
    cv=10,
    verbose=1,
    n_jobs=-1
)

print("Iniciando o Grid Search para encontrar os melhores hiperparâmetros...")
random_search.fit(X_train_red, y_train_red)

# Exibir os melhores hiperparâmetros encontrados
best_params = random_search.best_params_
print(f"Melhores hiperparâmetros encontrados: {best_params}")

# Treinar o modelo final com os melhores hiperparâmetros
best_model = random_search.best_estimator_
print("Treinando o modelo final com os melhores hiperparâmetros...")
best_model.fit(X_train_red, y_train_red)

# Avaliar o modelo no conjunto de teste
accuracy = best_model.score(X_test_red, y_test_red)
print(f"Acurácia do modelo no conjunto de teste: {accuracy:.4f}")

# Criar diretório para salvar o modelo otimizado
model_dir = "C:\\Users\\Arlison Gaspar\\Desktop\\testIA\\mushroom-classifier\\src\\"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Salvar o modelo treinado
model_path = os.path.join(model_dir, "svm_model_grid_search_reduced.pkl")
joblib.dump(best_model, model_path)
print(f"Modelo treinado salvo com sucesso em: {model_path}")