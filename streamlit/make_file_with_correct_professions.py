import pandas as pd
from spellchecker import SpellChecker
import streamlit as st

df = pd.read_csv('SF_Mandarin_dataset_ver3_csv.csv_rlkey=jkngphmwkoocingpxoga0uv2v', sep=';')

df.dropna(inplace=True)
df['Position'] = df['Position'].apply(lambda x: x.lower())
df['Position'] = df['Position'].str.replace(',', '')
df['Position'] = df['Position'].str.rstrip()

all_professions = list(df['Position'].unique())
all_single_professions = [prof for prof in all_professions if len(prof.split()) == 1]

# Получим синтакически-верные наименования профессий
spell = SpellChecker(language='RU')
all_known_professions = []

for word in all_single_professions:
    if spell.correction(word) == word and (set(' .-') & set(word)) == set():
        all_known_professions.append(word)


# Удалим из all_known_professions все подмножества, с которых начинаются другие профессии: ['груз', 'грузчик', 'погрузчик'] -> ['грузчик', 'погрузчик']
def filter_subsets(words):
    filtered_words = set(words)
    for word in words:
        for other_word in words:
            if word != other_word and other_word.startswith(word):
                filtered_words.discard(word)
                break
    return list(filtered_words)


all_known_professions = filter_subsets(all_known_professions)

file_path = "correct_professions.txt"

with open(file_path, "w", encoding="utf-8") as file:
    file.write(";".join(all_known_professions))
