from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carrega o modelo salvo
loaded_model = joblib.load("model/model.pickle")

# Carrega o encoder salvo
loaded_encoder = joblib.load("model/one_hot_encoder.pickle")


# Exemplo de GET
# /prever?nome=HR&marca=HYUNDAI&situação=Pane mecânica
@app.route("/prever", methods=["GET"])
def prever():
    if not request.args:
        return (
            "Sem parâmetros. Exemplo de chamada: "
            "/prever?nome=HR&marca=HYUNDAI&situação=Pane mecânica",
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


# Exemplo de POST
""" 
{
  "nome": "HR",
  "marca": "FORD",
  "situação": "Pane mecânica"
}
"""
@app.route("/prever", methods=["POST"])
def prever_post():
    # Verifica se a solicitação contém dados JSON
    if not request.is_json:
        return "A solicitação deve conter dados JSON", 400

    # Obtém os dados do corpo da solicitação
    data = request.json

    if data is None:
        return "Dados não informados", 400

    # Verifica se os campos necessários estão presentes nos dados
    required_fields = ["nome", "marca", "situação"]
    for field in required_fields:
        if field not in data:
            return f"Parâmetro {field.upper()} não informado", 400

    # Obtém os valores dos campos
    nome = data["nome"]
    marca = data["marca"]
    situacao = data["situação"]

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
    except Exception as e:
        return f"Erro ao codificar os dados: {str(e)}", 400

    # Faz a previsão com o modelo carregado
    predicted_type = loaded_model.predict(new_data_encoded)

    # Retornar o resultado como JSON
    return jsonify({"previsao": predicted_type[0]})


if __name__ == "__main__":
    print("Servidor Flask em execução")
    # Executar o aplicativo Flask
    app.run(debug=True)


# ENDPOINTS:
# GET /prever - retorna a previsão com base nos parâmetros informados
# POST /prever - retorna a previsão com base nos parâmetros informados

""" 
Exemplo de chamada GET:
http://localhost:5000/prever?nome=HR&marca=HYUNDAI&situação=Pane mecânica 
"""

""" 
Exemplo de chamada POST:
http://localhost:5000/prever 

JSON de entrada:
{
    "nome": "HR",
    "marca": "FORD",
    "situação": "Pane mecânica"
}
"""
