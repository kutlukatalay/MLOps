from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class modelSchema(BaseModel):
    gender:int
    SeniorCitizen:int
    Partner:int
    Dependents:int
    tenure:float
    PhoneService:int
    MultipleLines:int
    InternetSecurity:int
    OnlineSecurity:int
    OnlineBackup:int
    DeviceProtection:int
    TechSupport:int
    StreamingTV:int
    StreamingMovies:int
    Contract:int
    PaperlessBilling:int
    PaymentMethod:int
    MonthlyCharges:float
    TotalCharges:float

@app.get("/")
def hello():
    return {'mesaj':'modelin tahminleri gorulmektedir'}


@app.post("/predict/DecisionTreeClassifier/")
def predict_model(predict_value:modelSchema):
    load_model = pickle.load(open("final_model.sav","rb"))
    #print(predict_value)
    df = pd.DataFrame(
        [predict_value.dict().values()],
        columns = predict_value.dict().keys()
    )


    predict = load_model.predict(df)
    return {"Predict":int(predict[0])}