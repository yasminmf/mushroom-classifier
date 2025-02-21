import pandas as pd
from sklearn.preprocessing import LabelEncoder
def load_and_preprocess_data(file_path):
    """
    Carrega e pré-processa o dataset de cogumelos.
    - Converte atributos categóricos para valores numéricos usando LabelEncoder.
    - Retorna o DataFrame processado e os LabelEncoders usados.
    """
    # Definir nomes das colunas
    column_names = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
        "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
        "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
        "ring-number", "ring-type", "spore-print-color", "population", "habitat"
    ]
    
    # Carregar os dados
    df = pd.read_csv(file_path, header=None, names=column_names)

    # Criar dicionário para armazenar os codificadores LabelEncoder
    label_encoders = {}

    # Converter atributos categóricos para numéricos
    for column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le  # Guardamos o encoder para futuras conversões reversas
    return df, label_encoders

if __name__ == "__main__":
    # Testar a função
    df, _ = load_and_preprocess_data("C:\\Users\\Arlison Gaspar\\Desktop\\testIA\\mushroom-classifier\\data\\agaricus-lepiota.data")
    print(df.head())
