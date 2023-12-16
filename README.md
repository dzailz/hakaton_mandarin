markdown
# CreditProphet

Разработка ML-модели по предсказанию решения кредитной организации при оформлении займа/рассрочки на платформе [Mandarin.](https://mandarin.io/ru)

**Цель проекта:** Реализовать модель машинного обучения, которая сможет прогнозировать решение кредитной организации, вероятность одобрения займа, построенная на основе обезличенных данных, для повышения конверсии бизнеса. В качестве решения использовался алгоритм машинного обучения RandomForest c SMOTE (Synthetic Minority Oversampling Technique) — методом синтетической избыточной выборки меньшинства для каждого банка по отдельности.

## Начало работы
При подготовке проекта использовался Python 3.11.0

### Для запуска проекта локально:
```bash
git clone git@github.com:dzailz/hakaton_mandarin.git && cd hakaton_mandarin
```
Для запуска проекта в Docker (предварительно установить [docker](https://docs.docker.com/get-docker/)):
```bash
docker compose up -d --build
```
### Разработка
Для локальной разработки (предварительно установить [poetry](https://python-poetry.org/docs/#installation)) использовался Poetry 1.7.0:
```bash
poetry install --no-root --with=dev
```
## Использование
На основе анализа объема данных с заданными параметрами, ML-модель оценивает возможность получения займа клиентом в конкретных банках. Результаты модели могут быть использованы для принятия решений о выдаче займов и оптимизации конверсии бизнеса.

При запуске из докера сервер будет доступен по адресу [http://localhost:8888](http://localhost:8888) или [http://127.0.0.1:8888](http://127.0.0.1:8888)
Swagger доступен по адресу [http://localhost:8888/docs](http://localhost:8888/docs) или [http://127.0.0.1:8888/docs](http://127.0.0.1:8888/docs)
Проверить работоспособность можно с помощью команды:
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "birth_date": "1980-01-01",
  "education": "Высшее - специалист",
  "employment_status": "Работаю по найму полный рабочий день/служу",
  "value": "10 и более лет",
  "job_start_date": "2000-01-01",
  "position": "Manager",
  "month_profit": 1234567,
  "month_expense": 9876543,
  "gender": 1,
  "family_status": "Никогда в браке не состоял(а)",
  "child_count": 2,
  "snils": 1,
  "merch_code": 61,
  "loan_amount": 8765432,
  "loan_term": 12,
  "goods_category": "Mobile_devices"
}' http://localhost:8888/predict_bank_decision
```
## Пример ответа модели
```json
{
  "bank_a": {
    "prediction": "success",
    "probability": {
      "denied": 0.3,
      "success": 0.7
    }
  },
  "bank_b": {
    "prediction": "success",
    "probability": {
      "denied": 0.34,
      "success": 0.66
    }
  },
  "bank_c": {
    "prediction": "success",
    "probability": {
      "denied": 0.4,
      "success": 0.6
    }
  },
  "bank_d": {
    "prediction": "success",
    "probability": {
      "denied": 0.38,
      "success": 0.62
    }
  },
  "bank_e": {
    "prediction": "success",
    "probability": {
      "denied": 0.22,
      "success": 0.78
    }
  }
}

```
Модель базируется на scikit-learn [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). Для балансировки классов использовался [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html). Для оценки качества модели использовался roc_auc_score, precision_recall_curve, confusion_matrix.
ROC auc
![ROC auc](https://github.com/dzailz/hakaton_mandarin/blob/main/ROC_auc.jpeg)

Precision/recall
![Precision/recall](https://github.com/dzailz/hakaton_mandarin/blob/main/Precision_recall.jpeg)

Confusion matrix
![Confusion matrix](https://github.com/dzailz/hakaton_mandarin/blob/main/Confusion_matrix.jpeg)

## Команда
**Мандаринки->Новый год**
- Слободчикова Екатерина Валерьевна
  - Team lead, product manager
- Драгомирский Даглар Сарматович
  - NLP engineer, MLOps engineer
- Катин Владимир Викторович
  - Technical Lead, ML&MLOps engineer
- Орлов Александр Александрович
  - ML engineer
- Михайличенко Людмила Александровна
  - ML engineer, motivational speaker
- Зайцев Антон Александрович
  - ML engineer

Лицензия
Apache License, version 2.0


