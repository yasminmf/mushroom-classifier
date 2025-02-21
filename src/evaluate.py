import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import load_and_preprocess_data
from mpl_toolkits.mplot3d import Axes3D  # Importação necessária para gráficos 3D

# Carregar os dados pré-processados
file_path = "C:\\Users\\Arlison Gaspar\\Desktop\\testIA\\mushroom-classifier\\data\\agaricus-lepiota.data"
df, _ = load_and_preprocess_data(file_path)

# Separar features (X) e variável alvo (y)
selected_features = ["odor", "spore-print-color", "stalk-surface-below-ring", 
                     "stalk-color-above-ring", "habitat", "cap-color"]

X_reduced = df[selected_features]
y = df["class"]

# Modelo otimizado a ser avaliado
model_path = "mushroom-classifier\src\svm_model_grid_search_reduced.pkl"
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

# Função para plotar o hiperplano e as margens em 3D
def plot_3d_hyperplane_and_margins(model, X, y, features_to_plot):
    # Selecionar as features que serão plotadas
    X_3d = X[features_to_plot].values
    y = y.values

    # Calcular a média das features que não serão plotadas
    other_features_mean = X.drop(features_to_plot, axis=1).mean().values

    # Criar um grid para plotar
    x_min, x_max = X_3d[:, 0].min() - 1, X_3d[:, 0].max() + 1
    y_min, y_max = X_3d[:, 1].min() - 1, X_3d[:, 1].max() + 1
    z_min, z_max = X_3d[:, 2].min() - 1, X_3d[:, 2].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    zz = np.zeros_like(xx)

    # Preencher as outras features com a média
    other_features = np.tile(other_features_mean, (xx.size, 1))

    # Prever o valor para cada ponto no grid
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = np.array([xx[i, j], yy[i, j], zz[i, j]])
            point_full = np.hstack([point, other_features[0]])  # Adiciona as outras features
            zz[i, j] = model.decision_function([point_full])[0]

    # Plotar o gráfico 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotar o hiperplano
    ax.plot_surface(xx, yy, zz, alpha=0.5, cmap=plt.cm.coolwarm)

    # Plotar os pontos de dados
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

    ax.set_xlabel(features_to_plot[0])
    ax.set_ylabel(features_to_plot[1])
    ax.set_zlabel(features_to_plot[2])
    ax.set_title('Hiperplano e Margens do SVM em 3D')
    plt.show()

# Escolher as features que serão plotadas
features_to_plot = ["odor", "spore-print-color", "stalk-surface-below-ring"]  # Exemplo de features

# Plotar o hiperplano e as margens em 3D
plot_3d_hyperplane_and_margins(model, X_reduced, y, features_to_plot)