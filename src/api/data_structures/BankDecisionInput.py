from enum import Enum

from pydantic import BaseModel

from datetime import datetime


class BankDecisionInput(BaseModel):
    birth_date: str
    education: str
    employment_status: str
    value: str
    job_start_date: str
    position: str
    month_profit: int
    month_expense: int
    gender: int
    family_status: str
    child_count: int
    snils: int
    merch_code: int
    loan_amount: int
    loan_term: int
    goods_category: str



