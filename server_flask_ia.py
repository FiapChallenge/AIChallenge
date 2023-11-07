from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carrega o modelo salvo
loaded_model = joblib.load("model/model.pickle")

# Carrega o encoder salvo
loaded_encoder = joblib.load("model/one_hot_encoder.pickle")


# Rota para receber os dados e fazer previsões
@app.route("/prever", methods=["GET"])
def prever():
    if not request.args:
        return (
            "Sem parâmetros. Exemplo de chamada: "
            "/prever?NOME=HR&MARCA=HYUNDAI&SITUAÇÃO=Pane mecânica",
            400,
        )

    if not request.args.get("nome"):
        return "Parâmetro NOME não informado", 400

    if not request.args.get("marca"):
        return "Parâmetro MARCA não informado", 400

    if not request.args.get("situação"):
        return "Parâmetro SITUAÇÃO não informado", 400

    # Obter parâmetros da solicitação GET

    nome = request.args.get("nome")
    marca = request.args.get("marca")
    situacao = request.args.get("situação")

    new_data = pd.DataFrame(
        {
            "NOME": [nome],
            "MARCA": [marca],
            "SITUAÇÃO": [situacao],
        }
    )

    # Codifica os dados com o encoder carregado
    try:
        new_data_encoded = loaded_encoder.transform(new_data).toarray()
    except:
        return "Erro ao codificar os dados", 400

    # Faz a previsão com o modelo carregado
    predicted_type = loaded_model.predict(new_data_encoded)

    # Retornar o resultado como JSON
    return jsonify({"previsao": predicted_type[0]})


if __name__ == "__main__":
    print("Servidor Flask em execução")
    # Executar o aplicativo Flask
    app.run(debug=True)
