# Model name: CreditProphet

Development of an ML model for predicting the decision of a credit organization when applying for a loan/installment on the platform [Mandarin](https://mandarin.io/ru).

**Project goal:** Implement a machine learning model that can predict the decision of a credit organization, the probability of loan approval, based on anonymized data, to increase business conversion. The solution used was the RandomForest machine learning algorithm with SMOTE (Synthetic Minority Oversampling Technique) - a synthetic minority oversampling method for each bank separately.

## Getting Started

The project was prepared using Python 3.11.0

### To run the project locally

```bash
git clone git@github.com:dzailz/hakaton_mandarin.git && cd hakaton_mandarin
```

To run the project in Docker (pre-install [docker](https://docs.docker.com/get-docker/)):

```bash
docker compose up -d --build
```

### Development

For local development (pre-install [poetry](https://python-poetry.org/docs/#installation)), Poetry 1.7.0 was used:

```bash
poetry install --no-root --with=dev
```

## Usage

Based on the analysis of the volume of data with the specified parameters, the ML model assesses the possibility of the client receiving a loan in specific banks. The results of the model can be used to make decisions about issuing loans and optimizing business conversion.

When launched from docker, the server will be available at [http://localhost:8888](http://localhost:8888) or [http://127.0.0.1:8888](http://127.0.0.1:8888)

Swagger is available at [http://localhost:8888/docs](http://localhost:8888/docs) or [http://127.0.0.1:8888/docs](http://127.0.0.1:8888/docs)

You can check the performance with the command:

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

## Example of model response

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

The model is based on scikit-learn [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). For class balancing, [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) was used. To evaluate the quality of the model, roc_auc_score, precision_recall_curve, confusion_matrix were used.

ROC auc
![ROC auc](https://github.com/dzailz/hakaton_mandarin/blob/main/ROC_auc.jpeg)

Precision/recall
![Precision/recall](https://github.com/dzailz/hakaton_mandarin/blob/main/Precision_recall.jpeg)

Confusion matrix
![Confusion matrix](https://github.com/dzailz/hakaton_mandarin/blob/main/Confusion_matrix.jpeg)

## Team Mandarin->New Year

- Ekaterina Valeryevna Slobodchikova - Team lead, product manager
- Daglar Sarmatovich Dragomirsky - NLP engineer, MLOps engineer
- Vladimir Viktorovich Katin - Technical Lead, ML&MLOps engineer
- Alexander Alexandrovich Orlov - ML engineer
- Lyudmila Alexandrovna Mikhailichenko - ML engineer, motivational speaker
- Anton Alexandrovich Zaitsev - ML engineer

### License

[GPL v3](https://www.gnu.org/licenses/gpl-3.0.html)
