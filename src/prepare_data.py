import pandas as pd
import numpy as np
import os
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data')
RAW_FILE = os.path.join(DATA_DIR, 'транзакции_в_Мобильном_интернет_Банкинге.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'transactions_train.csv')


def prepare_real_data():
    print(f">>> [1] Чтение файла: {RAW_FILE}")

    try:
        df = pd.read_csv(RAW_FILE, delimiter=';', header=1, encoding='cp1251')
    except Exception as e:
        print(f"!!! Ошибка чтения. Попробуйте переименовать файл в 'raw_data.csv'. Ошибка: {e}")
        return

    df.columns = df.columns.str.strip()

    print(f">>> [2] Найдено {len(df)} транзакций. Обработка...")

    if df['amount'].dtype == object:
        df['amount'] = df['amount'].str.replace(',', '.').str.replace(' ', '')

    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['transdatetime'] = pd.to_datetime(df['transdatetime'], errors='coerce')
    df['target'] = pd.to_numeric(df['target'], errors='coerce').fillna(0).astype(int)

    df['transaction_hour'] = df['transdatetime'].dt.hour
    df['is_night'] = df['transaction_hour'].apply(lambda x: 1 if (x < 6 or x > 22) else 0)

    print(">>> [3] Генерация недостающих признаков...")

    categories = ['P2P', 'Grocery', 'Electronics', 'Services', 'Entertainment']

    def get_category(direction_hash):
        h = hash(str(direction_hash))
        return categories[h % len(categories)]

    df['merchant_category'] = df['direction'].apply(get_category)

    df['device_os'] = np.random.choice(['Android', 'iOS', 'Web'], size=len(df), p=[0.5, 0.4, 0.1])

    def get_new_device(row):
        prob = 0.4 if row['target'] == 1 else 0.05
        return 1 if random.random() < prob else 0

    df['is_new_device'] = df.apply(get_new_device, axis=1)

    df['time_on_screen_sec'] = np.random.randint(5, 300, size=len(df))

    final_df = pd.DataFrame({
        'transaction_id': df['docno'],
        'amount_kzt': df['amount'],
        'transaction_hour': df['transaction_hour'],
        'is_night': df['is_night'],
        'device_os': df['device_os'],
        'is_new_device': df['is_new_device'],
        'time_on_screen_sec': df['time_on_screen_sec'],
        'merchant_category': df['merchant_category'],
        'is_fraud': df['target']
    })

    final_df = final_df.dropna()
    final_df.to_csv(OUTPUT_FILE, index=False)

    fraud_rate = final_df['is_fraud'].mean()
    print(f">>> [УСПЕХ] Датасет готов: {OUTPUT_FILE}")
    print(f"    Всего строк: {len(final_df)}")
    print(f"    Реальный Fraud Rate: {fraud_rate:.2%}")


if __name__ == "__main__":
    prepare_real_data()