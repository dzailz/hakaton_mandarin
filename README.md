<body>

<p style="color: green; font-weight: bold;"><span>Цель проекта:</span> Реализовать модель машинного обучения, которая сможет прогнозировать решение кредитной организации, вероятность одобрения займа, построенная на основе обезличенных данных, для повышения конверсии бизнеса. В качестве решения использовался алгоритм машинного обучения RandomForest c SMOTE (Synthetic Minority Oversampling Technique) — методом синтетической избыточной выборки меньшинства для каждого банка по отдельности.</p>
<h2>Начало работы</h2>
<p>При подготовке проекта использовался Python 3.11.0</p>

<h3>Для запуска проекта локально:</h3>
<pre>
<code>git clone git@github.com:dzailz/hakaton_mandarin.git &amp;&amp; cd hakaton_mandarin</code>
</pre>

<p>Для запуска проекта в Docker (предварительно установить <a href="https://docs.docker.com/get-docker/">docker</a>):</p>
<pre>
<code>docker compose up -d --build</code>
</pre>

<h3>Разработка</h3>
<p>Для локальной разработки (предварительно установить <a href="https://python-poetry.org/docs/#installation">poetry</a>) использовался Poetry 1.7.0:</p>
<pre>
<code>poetry install --no-root --with=dev</code>
</pre>
<h2>Использование</h2>
<p>На основе анализа объема данных с заданными параметрами, ML-модель оценивает возможность получения займа клиентом в конкретных банках. Результаты модели могут быть использованы для принятия решений о выдаче займов и оптимизации конверсии бизнеса.</p>

<p>При запуске из докера сервер будет доступен по адресу <a href="http://localhost:8888">http://localhost:8888</a> или <a href="http://127.0.0.1:8888">http://127.0.0.1:8888</a></p>
<p>Swagger доступен по адресу <a href="http://localhost:8888/docs">http://localhost:8888/docs</a> или <a href="http://127.0.0.1:8888/docs">http://127.0.0.1:8888/docs</a></p>
<p>Проверить работоспособность можно с помощью команды:</p>
<pre><code>curl -X POST -H "Content-Type: application/json" -d '{
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
}' http://localhost:8888/predict_bank_decision</code></pre>

<h2>Пример ответа модели</h2>
<pre><code>{
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
}</code></pre>

<p>Модель базируется на scikit-learn <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">RandomForestClassifier</a>. Для балансировки классов использовался <a href="https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html">SMOTE</a>. Для оценки качества модели использовался roc_auc_score, precision_recall_curve, confusion_matrix.</p>

<p>ROC auc</p>
<img src="https://github.com/dzailz/hakaton_mandarin/blob/main/ROC_auc.jpeg" alt="ROC auc">

<p>Precision/recall</p>
<img src="https://github.com/dzailz/hakaton_mandarin/blob/main/Precision_recall.jpeg" alt="Precision/recall">

<p>Confusion matrix</p>
<img src="https://github.com/dzailz/hakaton_mandarin/blob/main/Confusion_matrix.jpeg" alt="Confusion matrix">

<h2><span style="color:green;">Команда</span>  <span style="color:orange;">Мандаринки->Новый год</span></h2>
<ul>
<li><span style="color:green;">Слободчикова Екатерина Валерьевна</span>
  - <span style="color:orange;">Team lead, product manager</span></li>
<li><span style="color:green;">Драгомирский Даглар Сарматович</span>
  - <span style="color:orange;">NLP engineer, MLOps engineer</span></li>
<li><span style="color:green;">Катин Владимир Викторович</span>
  - <span style="color:orange;">Technical Lead, ML&MLOps engineer</span></li>
<li><span style="color:green;">Орлов Александр Александрович</span>
  - <span style="color:orange;">ML engineer</span></li>
<li><span style="color:green;">Михайличенко Людмила Александровна</span>
  - <span style="color:orange;">ML engineer, motivational speaker</span></li>
<li><span style="color:green;">Зайцев Антон Александрович</span>
  - <span style="color:orange;">ML engineer</span></li>
</ul>
<h3>Лицензия</h3>
<p>[Apache License, version 2.0](https://www.apache.org/licenses/LICENSE-2.0.html)</p>

</body>

</html>
