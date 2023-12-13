from fastapi import FastAPI, Response, status

from src.api.data_structures.bank_decision import BankDecisionInput, BankDecisionOutput
from src.api.data_structures.errors import InvalidInputError
from src.api.data_structures.root import ModelAndTeamInfo
from src.domain.main import predict

app = FastAPI()


@app.get("/")
def read_root():
    return ModelAndTeamInfo()


@app.post("/predict_bank_decision", response_model=BankDecisionOutput | InvalidInputError, status_code=status.HTTP_200_OK)
def predict_bank_decision(data: BankDecisionInput, response: Response):
    try:
        result = predict(data=data.model_dump())
        # Return the prediction as a response
        return BankDecisionOutput(
            bank_a=result['bank_a'],
            bank_b=result['bank_b'],
            bank_c=result['bank_c'],
            bank_d=result['bank_d'],
            bank_e=result['bank_e']
        )
    except (ValueError, TypeError) as e:
        response.status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        return InvalidInputError(error=str(e))
