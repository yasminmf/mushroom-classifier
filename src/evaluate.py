import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import load_and_preprocess_data

# Carregar os dados pré-processados
file_path = "C:/Users/yasmi/INTELIGENCIA ARTIFICIAL/mushroom-classifier/data/agaricus-lepiota.data"
df, _ = load_and_preprocess_data(file_path)

# Separar features (X) e variável alvo (y)
selected_features = ["odor", "spore-print-color", "stalk-surface-below-ring", 
                     "stalk-color-above-ring", "habitat", "cap-color"]

X_reduced = df[selected_features]
y = df["class"]

# Modelo otimizado a ser avaliado
model_path = "../model/svm_model_gridsearch_reduced.pkl"
print(f"\nAvaliando modelo: {model_path}")
model = joblib.load(model_path)
X_test = X_reduced  # Avaliação usando as 6 features

# Fazer previsões
y_pred = model.predict(X_test)
accuracy = accuracy_score(y, y_pred)
print(f"Acurácia: {accuracy * 100:.2f}%")
print("Relatório de Classificação:")
print(classification_report(y, y_pred))

# Gerar matriz de confusão
conf_matrix = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Comestível", "Venenoso"], yticklabels=["Comestível", "Venenoso"])
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Modelo Reduzido Otimizado com GridSearch")
plt.show()
