from fastapi import FastAPI, Response, status

from src.api.data_structures.bank_decision import BankDecisionInput, BankDecisionOutput
from src.api.data_structures.errors import InvalidInputError
from src.api.data_structures.root import ModelAndTeamInfo
from src.domain.main import predict

app = FastAPI()


@app.get("/")
def read_root():
    """
    Endpoint to get the model and team information.

    This endpoint accepts a GET request and returns a ModelAndTeamInfo data structure.

    Returns:
        ModelAndTeamInfo: The model and team information.
    """
    return ModelAndTeamInfo()


@app.post("/predict_bank_decision", response_model=BankDecisionOutput | InvalidInputError, status_code=status.HTTP_200_OK)
def predict_bank_decision(data: BankDecisionInput, response: Response):
    """
    Endpoint to predict bank decision based on the provided data.

    This endpoint accepts a POST request with a JSON body containing the data required for the prediction.
    The data should be in the format of the BankDecisionInput data structure.

    If the prediction is successful, it returns a BankDecisionOutput data structure with the prediction results.
    If the input data is invalid, it returns an InvalidInputError data structure with an error message.

    Args:
        data (BankDecisionInput): The input data for the prediction. Should be a valid BankDecisionInput data structure.
        response (Response): The response object used to set the HTTP status code.

    Returns:
        Union[BankDecisionOutput, InvalidInputError]: The prediction results if the input data is valid, or an error message if the input data is invalid.

    Raises:
        ValueError: If the input data is not valid.
        TypeError: If the input data is not of the correct type.
    """
    try:
        result = predict(data=data.model_dump())
        # Return the prediction as a response
        return BankDecisionOutput(bank_a=result["bank_a"], bank_b=result["bank_b"], bank_c=result["bank_c"], bank_d=result["bank_d"], bank_e=result["bank_e"])
    except (ValueError, TypeError) as e:
        response.status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        return InvalidInputError(error=str(e))
