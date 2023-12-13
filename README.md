# Название проекта
<span style="color:orange;">Разработке ML-модели по предсказанию решения кредитной организации при оформлении займа методом бинарной классификации</span>  

<span style="color:green;">**Цель проекта:**</span>
Реализовать модель машинного обучения, которая сможет прогнозировать решение кредитной организации, выдать или отказать в выдаче займа, построенная на основе обезличенных данных, для повышения конверсии бизнеса.

В качестве решения использовался метод бинарной классификации, для каждого банка по отдельности.

## Начало работы
Для запуска проекта локально: 
```bibtex 
git clone git@github.com:dzailz/hakaton_mandarin.git
```
- войти в терминал   
- войти в папку с репозиторием    

войти в виртуальное окружение и установить необходимые библиотеки
```bibtex 
poetry shell
poetry install  
```

## Использование
На основе анализа объема данных с заданными параметрами, ML-модель оценивает возможность получения займа клиентом в конкретных банках. Результаты модели могут быть использованы для принятия решений о выдаче займов и оптимизации конверсии бизнеса.

## Примеры ответа модели
![image](https://github.com/dzailz/hakaton_mandarin)

```bibtex
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

##  <span style="color:green;">Команда</span>  <span style="color:orange;">Мандаринки->Новый год</span>
- <span style="color:green;">Слободчикова Екатерина Валерьевна</span> 
  - <span style="color:orange;">Team lead, product manager</span>
- <span style="color:green;">Драгомирский Даглар Сарматович</span>  
  - <span style="color:orange;">NLP engineer, MLOps engineer</span>
- <span style="color:green;">Катин Владимир Викторович</span> 
  - <span style="color:orange;">Technical Lead, ML&MLOps engineer</span>
- <span style="color:green;">Орлов Александр Александрович</span> 
  - <span style="color:orange;">ML engineer</span>
- <span style="color:green;">Михайличенко Людмила Александровна</span>  
  - <span style="color:orange;">ML engineer, motivational speaker</span> 
- <span style="color:green;">Зайцев Антон Александрович</span> 
  - <span style="color:orange;">ML engineer</span> 

## Лицензия
[Apache License, version 2.0](https://www.apache.org/licenses/LICENSE-2.0.html)
