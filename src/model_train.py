import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.svm import SVC
from preprocess import load_and_preprocess_data

# Carregar os dados pré-processados
file_path = "C:\\Users\\Arlison Gaspar\\Desktop\\testIA\\mushroom-classifier\\data\\agaricus-lepiota.data"
df, _ = load_and_preprocess_data(file_path)

# Definir as features reduzidas
selected_features = ["odor", "spore-print-color", "stalk-surface-below-ring", "stalk-color-above-ring", "habitat", "cap-color"]  

X_reduced = df[selected_features]
y = df["class"]

# 🔹 Validação Cruzada Antes do Treinamento
print("Executando validação cruzada antes do treinamento...")
model_baseline = SVC()
cv_scores_before = cross_val_score(model_baseline, X_reduced, y, cv=10, scoring="accuracy")

print(f"Acurácia média ANTES do treinamento: {cv_scores_before.mean():.4f}")
print(f"Desvio padrão ANTES do treinamento: {cv_scores_before.std():.4f}")

# Dividir os dados em treino e teste (70% treino, 30% teste)
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    X_reduced, y, test_size=0.3, random_state=42, stratify=y
)

# Definir o espaço de busca de hiperparâmetros para o Randomized Search
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# Configurar o Randomized Search com 10-fold cross-validation
random_search = RandomizedSearchCV(
    estimator=SVC(),
    param_distributions=param_grid,
    n_iter=90,
    scoring='accuracy',
    cv=10,
    verbose=1,
    n_jobs=-1
)

print("Iniciando o Randomized Search para encontrar os melhores hiperparâmetros...")
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

# 🔹 Validação Cruzada Depois do Treinamento
print("Executando validação cruzada depois do treinamento...")
cv_scores_after = cross_val_score(best_model, X_reduced, y, cv=10, scoring="accuracy")

print(f"Acurácia média DEPOIS do treinamento: {cv_scores_after.mean():.4f}")
print(f"Desvio padrão DEPOIS do treinamento: {cv_scores_after.std():.4f}")

# Criar diretório para salvar o modelo treinado
model_dir = "C:\\Users\\Arlison Gaspar\\Desktop\\testIA\\mushroom-classifier\\src\\"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Salvar o modelo treinado
model_path = os.path.join(model_dir, "svm_model_grid_search_reduced.pkl")
joblib.dump(best_model, model_path)
print(f"Modelo treinado salvo com sucesso em: {model_path}")

# Acessar os resultados da validação cruzada interna
cv_results = random_search.cv_results_

# Exibir informações relevantes da validação cruzada
print("Validação Cruzada dos hiperparametros - Resultados:")
for i in range(len(cv_results['mean_test_score'])):
    print(f"Combinacao {i + 1}:")
    print(f"  Hiperparâmetros: {cv_results['params'][i]}")
    print(f"  Média da Acurácia: {cv_results['mean_test_score'][i]:.4f}")
    print(f"  Desvio Padrão: {cv_results['std_test_score'][i]:.4f}")
    print(f"  Número de Divisões (Dobras) usadas: {cv_results['split0_test_score'][i]:.4f}, {cv_results['split1_test_score'][i]:.4f}, ...")  # Você pode adicionar mais splits conforme necessário
    print("=" * 50)

