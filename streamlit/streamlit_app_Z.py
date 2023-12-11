import streamlit as st
import os
import pandas as pd
from datetime import datetime

#ПОДГРУЗКА ФАЙЛА ДЛЯ position
path = os.path.dirname(__file__)
file_path = path + "/correct_professions.txt"
# Открытие файла для чтения
with open(file_path, "r", encoding="utf-8") as file:
    file_content = file.read()
# Разделение строки на список по точке с запятой
correct_professions = file_content.split(";")
# Отобразить выбранное значение после ввода
#st.write("Выбрано: ", selected_option)

#ОКНО ДЛЯ ВЫБОРА ВРУЧНУЮ ХАРАКТЕРИСТИК
def input_user_data():
    st.sidebar.header('Укажите Ваши данные (предоставьте пожалуйста достоверную информацию)')
    options = sorted(correct_professions)# Варианты для выбора с подсказками
    position = st.sidebar.selectbox("Выберите занимаемую Вами Должность", options, index=0, format_func=lambda x: f'{x}', key="position_select")
    birthdate = st.sidebar.date_input("Ваша дата рождения", 
                                      min_value=datetime(1930, 1, 1), max_value=datetime(2016, 12, 31), value=None, key=None)
    #age = st.sidebar.slider('Ваш Возраст', min_value=18, max_value=90, value=18, step=1)
    monthProfit = st.sidebar.slider('Ваш ежемесячный доход (в тыс. руб)',
                                    min_value=0, max_value=120, value=30, step=1)
    monthExpense = st.sidebar.slider('Ваш ежемесячный расход (в тыс. руб)',
                                    min_value=0, max_value=500, value=30, step=1)
    
    education = st.sidebar.selectbox('Образование',
                                    ('Высшее - специалист','Среднее профессиональное', 
                                     'Неоконченное высшее', 'Среднее', 'Бакалавр',
                                     'Несколько высших', 'Магистр','Неоконченное среднее',
                                     'MBA','Ученая степень'
                                     ))
    employment_status = st.sidebar.selectbox('Тип занятости',
                                      ('Работаю по найму полный рабочий день/служу',
                                       'Собственное дело','Работаю по найму неполный рабочий день',
                                       'Пенсионер','Студент','Декретный отпуск','Не работаю'))
    value = st.sidebar.selectbox('Общий стаж работы', #сделать слайдер?
                                    ('менее 4 месяцев','4 - 6 месяцев','6 месяцев - 1 год',
                                     '1 - 2 года','2 - 3 года','3 - 4 года','4 - 5 лет','5 - 6 лет',
                                     '6 - 7 лет','7 - 8 лет','8 - 9 лет','9 - 10 лет','10 и более лет'
                                     ))
    jobstartdate = st.sidebar.date_input("Дата трудоустройства на текущую работу", 
                                      min_value=datetime(1970, 1, 1), max_value=datetime(2023, 12, 12), value=None, key=None)
    #age_work = st.sidebar.slider('Количество лет проработанное на последней работе', min_value=0, max_value=40, value=6, step=1)
    gender = st.sidebar.selectbox('Пол', ('Мужчина', 'Женщина'))
    family_status = st.sidebar.selectbox('Семейный статус',#хочется сделать бинарным столбец
                                         ('Никогда в браке не состоял(а)','Женат / замужем',
                                          'Разведён / Разведена',
                                          'Гражданский брак / совместное проживание',
                                          'Вдовец / вдова'))
    сhildСount = st.sidebar.slider('Количество детей младше 18 лет',
                                   min_value=0, max_value=10, value=0, step=1)
    snils = st.sidebar.selectbox('У вас есть снилс?',('Да', "Нет"))
    merch_code = st.sidebar.slider('Определите мерч код Вашего магазина, (узнавайте у продавцов)',
        min_value=1, max_value=80, value=1, step=1)
    goods_category = st.sidebar.selectbox('Выберите категорию товара',
                                          ('Furniture','Fitness','Medical_services','Education',
                                          'Other','Travel','Mobile_devices'))
    loan_term = st.sidebar.selectbox('Выберите срок кредита(в месяцах)',
                                          ('7.5','12.5','17.5','22.5'))
    loan_amount = st.sidebar.slider('Сумма заказа(тыс.руб)',
                                      min_value=1, max_value=200, value=50, step=1)

# загрузка данных файлом
    uploaded_files = st.file_uploader("Хотите данные по нескольким людям? загрузите файл в формате csv", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("файл:", uploaded_file.name, "загружен")
    
#ПРОВЕРКА ДАННЫХ НА ТИП ДАННЫХ

    data = {
        'Position': position,
        'BirthDate': birthdate,
        'MonthProfit': int(monthProfit),
        'MonthExpense': int(monthExpense),
        'education': education,
        'employment status': employment_status,
        'Value': value,
        'JobStartDate': jobstartdate,
        'Gender': gender,#перевести в 0 и 1 
        'Family status': family_status,
        'ChildCount': int(сhildСount),
        'SNILS': snils, # перевести в 0 и 1
        'Merch_code': int(merch_code),
        'Goods_category': goods_category,
        'Loan_amount': int(loan_amount),
        'Loan_term': float(loan_term)
        }
    df = pd.DataFrame(data, index=[0])

    st.write('Проверьте введённые данные(все поля должны быть заполнены:')
    st.write('(при ошибке писать @Lucy_Mihko)')
    st.write(df[[col for col in df.columns[:5]]])
    st.write(df[[col for col in df.columns[5:9]]])
    st.write(df[[col for col in df.columns[9:12]]])
    st.write(df[[col for col in df.columns[12:]]])

    return df

def wirte_prediction():
    df = input_user_data()
    if st.button('Предсказать!'):
        #with st.spinner('Выполняется предсказание...')
            st.write('## Предсказание:')
            st.write( {df})


if __name__ == '__main__':
    wirte_prediction()