import streamlit as st
import os

path = os.path.dirname(__file__)
file_path = path + "/correct_professions.txt"

# Открытие файла для чтения
with open(file_path, "r", encoding="utf-8") as file:
    file_content = file.read()

# Разделение строки на список по точке с запятой
correct_professions = file_content.split(";")

# Варианты для выбора с подсказками
options = sorted(correct_professions)

# Виджет selectbox с возможностью ввода и подсказками
selected_option = st.selectbox("Должность", options, index=0, format_func=lambda x: f'{x}', key="fruit_select")

# Отобразить выбранное значение после ввода
st.write("Выбрано: ", selected_option)