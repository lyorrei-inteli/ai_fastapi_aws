from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from pycaret.classification import load_model, predict_model

# Carregando o modelo salvo com PyCaret
model = load_model("./model/final_model")

app = FastAPI(
    title="Modelo de Previsão",
    description="Um API simples para fazer previsões usando um modelo treinado com PyCaret.",
    version="1.0.0",
)

# Definindo o formato do input usando Pydantic
class InputData(BaseModel):
    car: float
    bicycle: float
    trucks: float
    motorbike: float
    bus: float
    others: float
    animals: float
    special_cargos: float
    tractors: float
    utilities: float
    unharmed: float
    slight_injury: float
    moderate_injury: float
    serious_injury: float
    month: int
    year: int
    dayofweek: int
    road_info_Descending_Curve: int
    road_info_East_Going: int
    road_info_North_Going: int
    road_info_Other: int
    road_info_South_Going: int
    road_info_Top_Curve: int
    road_info_West_Going: int
    accident_place_Autopista_Fernão_Dias: int
    accident_place_Autopista_Fluminense: int
    accident_place_Autopista_Litoral_Sul: int
    accident_place_Autopista_Planalto_Sul: int
    accident_place_Autopista_Regis_Bittencourt: int
    accident_place_Concebra: int
    accident_place_Concepa: int
    accident_place_Concer: int
    accident_place_Cro: int
    accident_place_Crt: int
    accident_place_ECO050: int
    accident_place_ECO101: int
    accident_place_Ecoponte: int
    accident_place_Ecoriominas: int
    accident_place_Ecosul: int
    accident_place_Ecovias_do_Araguaia: int
    accident_place_Ecovias_do_Cerrado: int
    accident_place_MSVIA: int
    accident_place_Novadutra: int
    accident_place_RIOSP: int
    accident_place_Rodovia_do_Aço: int
    accident_place_Transbrasiliana: int
    accident_place_VIA040: int
    accident_place_Via_Bahia: int
    accident_place_Via_Brasil: int
    accident_place_Via_Costeira: int
    accident_place_Via_Sul: int

column_mapping = {
    "road_info_Descending_Curve": "road_info_Descending Curve",
    "road_info_East_Going": "road_info_East Going",
    "road_info_North_Going": "road_info_North Going",
    "road_info_South_Going": "road_info_South Going",
    "road_info_Top_Curve": "road_info_Top Curve",
    "road_info_West_Going": "road_info_West Going",
    "accident_place_Autopista_Fernão_Dias": "accident_place_Autopista Fernão Dias",
    "accident_place_Autopista_Fluminense": "accident_place_Autopista Fluminense",
    "accident_place_Autopista_Litoral_Sul": "accident_place_Autopista Litoral Sul",
    "accident_place_Autopista_Planalto_Sul": "accident_place_Autopista Planalto Sul",
    "accident_place_Autopista_Regis_Bittencourt": "accident_place_Autopista Regis Bittencourt",
    "accident_place_Ecovias_do_Araguaia": "accident_place_Ecovias do Araguaia",
    "accident_place_Ecovias_do_Cerrado": "accident_place_Ecovias do Cerrado",
    "accident_place_Rodovia_do_Aço": "accident_place_Rodovia do Aço",
    "accident_place_Via_Bahia": "accident_place_Via Bahia",
    "accident_place_Via_Brasil": "accident_place_Via Brasil",
    "accident_place_Via_Costeira": "accident_place_Via Costeira",
    "accident_place_Via_Sul": "accident_place_Via Sul"
}

class OutputData(BaseModel):
    prediction: float
    result: str

@app.post("/predict/", response_model=OutputData)
async def predict(data: InputData):
    try:
        # Convertendo o input para DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Convertendo colunas para o formato esperado pelo modelo
        input_data = input_data.rename(columns=column_mapping)

        # Fazendo previsão usando o modelo
        prediction = predict_model(model, data=input_data)

        # Retornando o resultado
        result = prediction['prediction_label'].iloc[0]
        return {"prediction": result, "result": "Acidente severo" if result == 1 else "Acidente não severo"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
