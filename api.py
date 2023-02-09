#from source.model import get_model
from typing import Dict
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from bertmodel import DiaTransModel, get_model

app = FastAPI ()

class DiagnosisRequest(BaseModel):
    text: str

class DiagnosisResponse(BaseModel):
    #ICD9_Codes: list
    #probabilities: Dict[str, float]
    diagnosis: list
    

@app.post("/predict", response_model=DiagnosisResponse)
def predict (request: DiagnosisRequest, model: DiaTransModel = Depends(get_model)):
    my_diagnoses_response = model.predict(request.text)
    return DiagnosisResponse( diagnosis = my_diagnoses_response
        #ICD9_Codes = ['4280', 'OTHER', '42731', '4019', '51881'],
        #probabilities = {"4280"= 0.10577,"OTHER"= 0.9956, "42731"= 0.1126,"4019" = 0.3217},
        #probabilities = {"4280": 0.10577,"OTHER": 0.9956},
        #diagnosis = ['Acute respiratory failure','OTHER','Congestive heart failure, unspecified', 'Atrial fibrillation']
    )