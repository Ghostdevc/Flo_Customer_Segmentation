import numpy as np
import pandas as pd
from datetime import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

def load_dataset():
    return pd.read_csv('Research/Dataset/flo_data_20k.csv')
    
df = load_dataset()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print('##################### Unique Values #####################')
    print(dataframe.nunique())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

date_cols = [col for col in df.columns if 'date' in col]

df[date_cols] = df[date_cols].apply(lambda x: pd.to_datetime(x))


df['recency'] = df['recency'] = (pd.to_datetime(datetime.now().date()) - df['last_order_date']).dt.days

df['frequency'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']

df['monetary'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']
