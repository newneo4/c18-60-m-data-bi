import json
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

#Inicializamos la aplicacion FastApi
app = FastAPI()

#Se define que la app solo va a revisir variables enteras en el input
class ModelInput(BaseModel):
    No_Checkout_Confirmed: int
    No_Checkout_Initiated: int
    No_Customer_Login: int
    Session_Activity_Count: int

#Se carga el modelo guardado anteriormente
customer_model = pickle.load(open('trained_model/CartAbandoned_model.pkl','rb'))
#Definir un punto final para la predicción del modelo usando el método POST
@app.post('/model_prediction')
def model_pred(input_parameters: ModelInput):

    No_c_c = input_parameters.No_Checkout_Confirmed
    No_c_i = input_parameters.No_Checkout_Initiated
    No_C_l = input_parameters.No_Customer_Login
    s_a_c = input_parameters.Session_Activity_Count
    #Preparar los datos de entrada para el modelo en formato de lista.
    input_list = [No_c_c,No_c_i,No_C_l, s_a_c]
    #Hacer una predicción usando el modelo cargado.
    prediction = customer_model.predict([input_list])
    #Interpretar el resultado de la predicción.
    if prediction[0] == 0:
        result = {"label": 0,
                  "result": "La persona no es propensa a abandonar el carrito"}
        return result
    else:
        result = {"label": 1,
                  "result": "La persona es propensa a abandonar el carrito"}
        return result