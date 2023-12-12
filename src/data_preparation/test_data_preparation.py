import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from scipy.stats import anderson
from pandas import DataFrame
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

current_date = pd.Timestamp('2023-11-11')

class DataPreparation_new:
    
   def __init__(self, dataframe):
        self.df = dataframe
        
   def remove_nan_and_duplicates(self):
        self.df = self.df[self.df.SkillFactory_Id.notna()]
        self.df = self.df.dropna().drop_duplicates()
        
   def process_text(self, text):
        text = text.strip()
        text = text.lower()
        text = text.replace('-', '')
        return text
    
   def apply_text_processing(self, column_name):
        self.df['Position'] = self.df['Position'].apply(self.process_text)
        
   def calculate_mean_and_fill_na(self, group_column, value_column):
        self.df[group_column] = self.df.groupby(group_column)[value_column].transform('mean').fillna(0).astype(int)
        
   def map_gender(self):
        self.df['Gender'] = pd.to_numeric(self.df['Gender'], errors='coerce').fillna(0).astype(int)
        self.df['Gender'] = self.df['Gender'].apply(lambda x: 0 if x == 'female' else 1 if x == 'male' else x)
        
   def map_family_status(self):
        self.df['Family status'] = self.df['Family status'].map({'Никогда в браке не состоял(а)': 0, 'Разведён / Разведена': 0, 'Гражданский брак / совместное проживание': 0, 'Вдовец / вдова': 0, 'Женат / замужем': 1})
        
   def replace_child_count(self):
        self.df['ChildCount'] = self.df['ChildCount'].replace({0.0: 0, 1.0: 1, 2.0: 1, 3.0: 1, 4.0: 1, 5.0: 1})
        
   def label_encode_goods_category(self):
        self.df['Goods_category'] = pd.factorize(self.df['Goods_category'])[0]
        
   def label_encode_education_employment_value(self):
        label_encoder = LabelEncoder()
        self.df['education'] = label_encoder.fit_transform(self.df['education'])
        self.df['employment status'] = label_encoder.fit_transform(self.df['employment status'])
        self.df['Value'] = label_encoder.fit_transform(self.df['Value'])
        
   def remove_rows_if_profit_less_than_expense(self):
        initial_rows = len(self.df)
        mask = self.df['MonthProfit'] < self.df['MonthExpense']
        if mask.any():
            self.df.drop(self.df[mask].index, inplace=True)
            final_rows = len(self.df)
            if final_rows < initial_rows:
                print("Ошибка: Условие MonthProfit < MonthExpense было выполнено и соответствующие строки были удалены.")
            else:
                print("Успешно: Ни одна строка не была удалена, так как условие не было выполнено.")
        else:
            print("Успешно: Ни одна строка не удовлетворяет условию MonthProfit < MonthExpense.")
            
   def add_zeros_to_3_digits(self, num):
        if len(str(num)) == 2:
            return int(str(num) + '000')
        else:
            return num
        
   def preprocess_dataframe(self):
        self.remove_rows_if_profit_less_than_expense()
        self.df['MonthProfit'] = self.df['MonthProfit'].apply(self.add_zeros_to_3_digits)
        self.df['MonthExpense'] = self.df['MonthExpense'].apply(self.add_zeros_to_3_digits)
        self.df['MonthExpense'] = self.df['MonthExpense'].replace(0, 14375)
        self.df['Loan_pay_month'] = (self.df['Loan_amount'] / self.df['Loan_term']).astype(int)
        
   def calculate_age(self, current_date):
        self.df['JobStartDate'] = pd.to_datetime(self.df['JobStartDate'])
        self.df['work_age'] = ((current_date - self.df['JobStartDate']).dt.days / 365.25).round(1)
        self.df.drop('JobStartDate', axis=1, inplace=True)
        
   
   def replace_values_with_regex_and_fill_mean(self, column_name, replacements):
       self.df[column_name] = self.df[column_name].replace(replacements, regex=True)
       self.df[column_name] = self.df.groupby(column_name)['MonthProfit'].transform('mean').fillna(0).astype(int)
        
# Replacing values in the 'Position' column using regex
replacements = ({

                    "\b(hr)+(менеджер|manager)?\b" : "hr менеджер",
                    "\b(pr-?)+(менеджер|manager)?\b" : "pr менеджер",
                    "\b(smm\s\смм?)+(специалист|менеджер)?\b" : "smm специалист",
                    "аген(т)?(\sпо\sнедвижимости)?" : "агент по недвижимости",
                    "^(в)?(а)?дми+нистратор[нрс]?\s?(/\sавтомойщик)?|адми?н?|адм$" : "администратор",
                    "^(аппа)+.*" : "аппаратчик",
                    "^(акт)+.*" : "актер",
                    "ас(с)?истент.*" : "ассистент",
                    "^(бар)+.*" : "бармен/бариста",
                    "^(борт)+.*" : "бортпроводник",
                    "^(бриг)+.*" : "бригадир",
                    "^(бух)+.*" : "бухгалтер",
                    "^(вед)+.*" : "ведущий специалист",
                    "^(ветери)+.*" : "ветеринар",
                    "^(водит)+.*" : "водитель",
                    "^(вос)+.*" : "воспитатель",
                    "^(вра)+.*" : "врач",
                    "^газо(резчик|(электро)?сварщик)+.*" : "газоэлектросварщик",
                    "^(гальва)+.*" : "гальваник",
                    "^(ген)+.*" : "генеральный директор",
                    "^(главный)+.*" : "главный специалист",
                    "^(груз)+.*" : "грузчик",
                    "^(дежурн(ый|ая))+.*" : "дежурный/дежурная",
                    "^(дизай)+.*" : "дизайнер",
                    "^(дир|владелец|учредитель)+.*" : "директор",
                    "^(диспет)+.*" : "диспетчер",
                    "д(н)?елопроизводитель" : "делопроизводитель",
                    "^(заведующ)+.*" : "заведующий",
                    "^(закро)+.*" : "закройщик",
                    "^(зам)+.*" : "заместитель директора",
                    "^(ип|инд|пре)+.*" : "индивидуальный предприниматель",
                    "^(инж)+.*" : "инженер",
                    "^(инс)+.*" : "инспектор",
                    "^(касс)+.*" : "кассир",
                    "^(кла)+.*" : "кладовщик",
                    "^(км|комъ)+.*" : "комьюнити-менеджер",
                    "^(командир)+.*" : "командир",
                    "^(консу)+.*" : "консультант",
                    "^контр(о|а)лер+.*" : "контролер",
                    "^косм+.*" : "косметолог",
                    "^лаборант+.*" : "лаборант",
                    "^мал+.*" : "маляр",
                    "^мар+.*" : "маркетолог",
                    "^маст+.*" : "мастер",
                    "^маши+.*" : "машинист",
                    "^мед+.*" : "медбрат/медсестра",
                    "^м(е|э|н)н+.*" : "менеджер",
                    "^мерч+.*" : "мерчендайзер",
                    "^меха+.*" : "механик",
                    "^младший+.*" : "младший сотрудник",
                    "^монт+.*" : "монтажник/монтер",
                    "^музыкальный+.*" : "музыкальный руководитель",
                    "^нала+.*" : "наладчик оборудования",
                    "^намот+.*" : "намотчик",
                    "^нача+.*" : "начальник",
                    "^научный+.*" : "научный сотрудник",
                    "^опера+.*" : "оператор",
                    "^организатор+.*" : "организатор",
                    "^отдел+.*" : "отделочник",
                    "^офи+.*" : "официант",
                    "^охр+.*" : "охранник",
                    "^пари+.*" : "парикмахер",
                    "^педагог+.*" : "педагог",
                    "^пен+.*" : "пентестер",
                    "^персональный+.*" : "персональный менеджер",
                    "^пова+.*" : "повар",
                    "^подсобный+.*" : "подсобный рабочий",
                    "^полице+.*" : "полицейский",
                    "^помо+.*" : "помощник",
                    # "^пре+.*" : "предприниматель",
                    "^председатель+.*" : "председатель",
                    "^предста+.*" : "представитель",
                    "^препо+.*" : "преподаватель",
                    "^приемщик+.*" : "приемщик",
                    "^прог+.*" : "программист",
                    "^прод+.*" : "продавец",
                    "^продюс(с)?ер" : "продюсер",
                    "^(проектный|проджект)?\s+менеджер" : "проджект менеджер",
                    "^(прово)+.*" : "проводник",
                    "^(прора)+.*" : "прораб",
                    "^(прора)+.*" : "прораб",
                    "^(психолог)+.*" : "психолог",
                    "^(рабо)+.*" : "рабочий",
                    "^(разнораб)+.*" : "разнорабочий",
                    "^(разработчик|программист)+.*" : "разработчик",
                    "^(реж)+.*" : "режиссер",
                    "^(р(и)?(е|э)+)+.*" : "риелтор",
                    "^(рук)+.*" : "руководитель отдела",
                    "^(само)+.*" : "самозанятый",
                    "^(санитар)+.*" : "санитар",
                    "^(сборщик)+.*" : "сборщик",
                    "^(сва)+.*" : "сварщик",
                    "^(секре)+.*" : "секретарь",
                    "^(с(л|о)е)+.*" : "слесарь",
                    "^(смотр)+.*" : "смотритель",
                    "^(снабже)+.*" : "снабженец",
                    "^(сот)+.*" : "сотрудник",
                    "^(спец|spec|промы)+.*" : "специалист",
                    "^(стано)+.*" : "станочник",
                    "^(стар)+.*" : "старший специалист",
                    "^(сто)+.*" : "столяр",
                    "^(судебный)+.*" : "судебный пристав",
                    "^(такси)+.*" : "таксист",
                    "^(терми)+.*" : "термист",
                    "^(телеф)+.*" : "телефонист",
                    "^(тех)+.*" : "техник",
                    "^(товар)+.*" : "товаровед",
                    "^(то)+.*" : "токарь",
                    "^(тренер)+.*" : "тренер",
                    "^(убор)+.*" : "уборщик",
                    "^(универсал)+.*" : "универсальный продавец",
                    "^(управ|упров|управ|упров)+.*" : "управляющий",
                    "^(установщик)+.*" : "установщик",
                    "^(учa)+.*" : "участковый",
                    "^(учи|учи|пед)+.*" : "учитель",
                    "^(фарм)+.*" : "фармацевт",
                    "^(фасов)+.*" : "фасовщик",
                    "^(фин)+.*" : "финансовый директор",
                    "^(фото)+.*" : "фотограф",
                    "^(фриланс)+.*" : "фрилансер",
                    "^(хор)+.*" : "хормейстер",
                    "^(худ)+.*" : "художник",
                    "^(шве)+.*" : "швея",
                    "^(шеф)+.*" : "шеф-повар",
                    "^(экономист)+.*" : "экономист",
                    "^(эксп)+.*" : "экспедитор",
                    "^(эксперт)+.*" : "эксперт",
                    "^(эл)+.*" : "электрик/электрогазосварщик",
                    "^(юри)+.*" : "юрист" }, regex=True)

   def process_gender_column(self):
      df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce').fillna(0).astype(int)
      df['Gender'] = df['Gender'].apply(lambda x: 0 if x == 'female' else 1 if x == 'male' else x)
      return df
# Mapping 'Family status' to binary values
   def map_family_status_column(self):
      conditions = df['Family status'].isin(['Никогда в браке не состоял(а)', 'Разведён / Разведена', 'Гражданский брак / совместное проживание', 'Вдовец / вдова'])
      df.loc[conditions, 'Family status'] = 0
      df.loc[df['Family status'] == 'Женат / замужем', 'Family status'] = 1
      return df

# Replacing values in 'ChildCount'
   def process_child_count_column(self):
      df['ChildCount'] = df['ChildCount'].replace({0.0: 0, 1.0: 1, 2.0: 1, 3.0: 1, 4.0: 1, 5.0: 1})
      return df

# Encoding 'Goods_category' using label encoding
   def process_goods_category_column(self):
      df['Goods_category'] = pd.factorize(df['Goods_category'])[0]
      total_loan_amount_per_category = df.groupby('Goods_category')['Loan_amount'].transform('sum')
      unique_customers_per_category = df.groupby('Goods_category')['SkillFactory_Id'].transform('nunique')
      average_loan_amount_per_category = total_loan_amount_per_category / unique_customers_per_category
      df['Goods_category'] = average_loan_amount_per_category
      return df

# Encoding 'education', 'employment status', and 'Value' using LabelEncoder
   def label_encode_education_employment_value(self):
      label_encoder = LabelEncoder()
      df['education'] = label_encoder.fit_transform(df['education'])
      df['employment status'] = label_encoder.fit_transform(df['employment status'])
      df['Value'] = label_encoder.fit_transform(df['Value'])
      return df


   def remove_rows_if_profit_less_than_expense(self):
      initial_rows = len(dataframe)
      mask = dataframe['MonthProfit'] < dataframe['MonthExpense']
      if mask.any():
         dataframe.drop(dataframe[mask].index, inplace=True)
         final_rows = len(dataframe)
         if final_rows < initial_rows:
               print("Ошибка: Условие MonthProfit < MonthExpense было выполнено и соответствующие строки были удалены.")
         else:
               print("Успешно: Ни одна строка не была удалена, так как условие не было выполнено.")
      else:
         print("Успешно: Ни одна строка не удовлетворяет условию MonthProfit < MonthExpense.")

   def add_zeros_to_3_digits(self, column_name):
      self.df[column_name] = self.df[column_name].apply(lambda num: int(str(num) + '000') 
                                                         if len(str(num)) == 2 else num)

   def preprocess_dataframe(self):
      dataframe['MonthProfit'] = dataframe['MonthProfit'].apply(add_zeros_to_3_digits)
      dataframe['MonthExpense'] = dataframe['MonthExpense'].apply(add_zeros_to_3_digits)
      dataframe['MonthExpense'] = dataframe['MonthExpense'].replace(0, 14375)
      dataframe['Loan_pay_month'] = (dataframe['Loan_amount'] / dataframe['Loan_term']).astype(int)
      return dataframe

   def calculate_age(self, current_date):
      df['JobStartDate'] = pd.to_datetime(df['JobStartDate'])
      df['work_age'] = ((current_date - df['JobStartDate']).dt.days / 365.25).round(1)
      df.drop('JobStartDate', axis=1, inplace=True)

      df['BirthDate'] = pd.to_datetime(df['BirthDate'])
      df['age'] = ((current_date - df['BirthDate']).dt.days / 365.25).round(1)
      df.drop('BirthDate', axis=1, inplace=True)


   def remove_outliers_all_numeric_with_condition(self, df: DataFrame | None = None, condition_value: str = 'success',
                                                   multiplier: float = 1.5, is_new: bool = True):
        """
      Remove outliers from all numeric columns in a DataFrame using the IQR method, based on a specific condition.

      Parameters:
        - condition_value: Value in the condition_column for which outliers will be removed (default is 'success')
        - multiplier: Multiplier to control the range of the IQR (default is 1.5)

        Returns:
        - DataFrame with outliers removed based on the specified condition
        """
        # Select numeric columns
        if df is None:
            df = self.df
        numeric_columns = df.select_dtypes(include='number').columns

        # Create a copy of the original DataFrame
        df_no_outliers = df.copy()

        # Iterate through numeric columns and remove outliers using IQR method with the specified condition
        for column in numeric_columns:
            # Calculate IQR only for rows where the condition is met
            condition_mask = (df_no_outliers[self.target_bank_col] == condition_value)
            q1 = df_no_outliers.loc[condition_mask, column].quantile(0.25)
            q3 = df_no_outliers.loc[condition_mask, column].quantile(0.75)
            iqr = q3 - q1

            # Define outlier bounds
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            # Remove outliers only for rows where the condition is met and the condition value is in outliers
            outliers_mask = condition_mask & (df_no_outliers[column] < lower_bound) | (
                    df_no_outliers[column] > upper_bound)
            df_no_outliers = df_no_outliers[~outliers_mask]
        if is_new:
            self.df_no_outliers = df_no_outliers
        else:
            self.df = df_no_outliers
 
df = remove_outliers_all_numeric_with_condition(df, condition_column='BankA_decision', condition_value='success', multiplier=1.5)

   def filter_decisions(df, columns):
    for col in columns:
        df = df[df[col] != 'error']
    return df

   def encode_decision(df, column_name):
    encoding_map = {'denied': 0, 'success': 1}  # Обновленное кодирование
    df[column_name] = df[column_name].map(encoding_map)
    return df

    def save_df(self, path: str, index: bool = False, compression='gzip'):
        """
        Save the data frame to parquet
        """
        self.df.to_parquet(path, index=index, compression=compression)
        

if __name__ == '__main__':
    from os import path
    banks = [f'bank_{bank}_decision' for bank in ['a', 'b', 'c', 'd', 'e']]

    for i in ['a', 'b', 'c', 'd', 'e']:
        banks_to_drop = banks.copy()
        banks_to_drop.remove(f'bank_{i}_decision')

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        df_path = path.join(path.dirname(__file__), '../../data/datasets/SF_Mandarin_dataset_ver3.csv')

        df = pd.read_csv(df_path, sep=';', index_col=0)

        bank_a = DataPreparation_new(
            df=df,
            to_drop_columns=banks_to_drop,
            target_bank_col=[i for i in banks if i not in banks_to_drop].pop()
        )
        
         bank_a=remove_nan_and_duplicates()      
         bank_a=process_text(column='Position')
         bank_a=apply_text_processing(column='Position')     
         bank_a=calculate_mean_and_fill_na()       
         bank_a=map_gender():     
         bank_a=map_family_status()
         bank_a=replace_child_count():  
         bank_a=label_encode_goods_category()
         bank_a=label_encode_education_employment_value()
         bank_a=remove_rows_if_profit_less_than_expense()
         bank_a=preprocess_dataframe()
         bank_a=replace_values_with_regex_and_fill_mean(column='Position', replacements)
         bank_a=process_gender_column
         bank_a=map_family_status_column()
         bank_a=replace_child_count(df)
         bank_a=label_encode_goods_category()
         bank_a=remove_rows_if_profit_less_than_expense()
         bank_a=label_encode_education_employment_value()
         bank_a=remove_rows_if_profit_less_than_expense
         bank_a=add_zeros_to_3_digits(self, num)
         bank_a=preprocess_dataframe()
         bank_a=calculate_work_age(current_date)
         bank_a=calculate_age(current_date)
         bank_a=remove_outliers_all_numeric_with_condition
         bank_a=filter_decisions(columns)
         bank_a=encode_decision(column_name)
         bank_a.remove_outliers_all_numeric_with_condition(is_new=False)
         # save_path = path.join(path.dirname(__file__), f'../../data/datasets/bank_{i}_ohe_norm.parquet')
         # bank_a.save_df(save_path)
         

