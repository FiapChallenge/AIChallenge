import joblib
import pandas as pd

# Carrega o modelo salvo
loaded_model = joblib.load("model/model.pickle")

# Carrega o encoder salvo
loaded_encoder = joblib.load("model/one_hot_encoder.pickle")

# Exemplo de dados para prever o tipo de guincho
new_data = pd.DataFrame(
    {
        "NOME": ["HR"],
        "MARCA": ["HYUNDAI"],
        "SITUAÇÃO": ["Pane mecânica"],
    }
)

# Codifica os dados com o encoder carregado
new_data_encoded = loaded_encoder.transform(new_data).toarray()

# Faz a previsão com o modelo carregado
predicted_type = loaded_model.predict(new_data_encoded)

print(f"Tipo de guincho previsto: {predicted_type[0]}")
