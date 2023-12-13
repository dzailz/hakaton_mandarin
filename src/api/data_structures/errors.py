from pydantic import BaseModel


class InvalidInputError(BaseModel):
    error: str = "Invalid input"
