import streamlit as st
import os
import pandas as pd

path = os.path.dirname(__file__)
file_path = path + "/correct_professions.txt"

# Открытие файла для чтения
with open(file_path, "r", encoding="utf-8") as file:
    file_content = file.read()
# Разделение строки на список по точке с запятой
correct_professions = file_content.split(";")
# Отобразить выбранное значение после ввода
#st.write("Выбрано: ", selected_option)

def input_user_data():
    st.sidebar.header('Укажите Ваши данные (предоставьте пожалуйста достоверную информацию)')
    options = sorted(correct_professions)# Варианты для выбора с подсказками
    position = st.sidebar.selectbox("Выберите занимаемую Вами Должность", options, index=0, format_func=lambda x: f'{x}', key="position_select")
    age = st.sidebar.slider('Ваш Возраст', min_value=18, max_value=90, value=18, step=1)
    monthProfit = st.sidebar.slider('Ваш ежемесячный доход (в тыс. руб)',
                                    min_value=0, max_value=120, value=30, step=1)
    monthExpense = st.sidebar.slider('Ваш ежемесячный доход (в тыс. руб)',
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
                                    ('9 - 10 лет','1 - 2 года','10 и более лет','2 - 3 года',
                                     '7 - 8 лет','3 - 4 года','5 - 6 лет','4 - 5 лет','6 - 7 лет','6 месяцев - 1 год',
                                     '4 - 6 месяцев','8 - 9 лет','менее 4 месяцев'
                                     ))
    age_work = st.sidebar.slider('Количество лет проработанное на последней работе', min_value=0, max_value=40, value=6, step=1)
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
    


def wirte_prediction():
    df = input_user_data()
    if st.button('Предсказать!'):
        with st.spinner('Выполняется предсказание...'):
            #pred, prob = make_prediction(df)
            st.write('## Предсказание:')
            st.write( {df})


if __name__ == '__main__':
    wirte_prediction()