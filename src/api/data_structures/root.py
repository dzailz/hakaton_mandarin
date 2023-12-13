from pydantic import BaseModel


class ModelAndTeamInfo(BaseModel):
    model_name: str = "Random Forest"
    team_name: str = "Мандаринки->НовыйГод"
    description: str = "Прогнозирование вероятности одобрения заявки на кредит"
    team_leader: str = "Слободчикова Екатерина Валерьевна"
    mlops_engineer: str = "Катин Владимир Викторович"
    nlp_engineer: str = "Драгомирский Даглар Сарматович"
    motivational_speaker: str = "Михайличенко Людмила Александровна",
    ml_engineer: str = ["Орлов Александр Александрович", "Зайцев Антон Александрович"]
