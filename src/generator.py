print(">>> [1] Начало работы скрипта...")  # Если вы не увидите это, значит вы запускаете не тот файл

import os

print(">>> [2] Импорт системных библиотек прошел")

try:
    import pandas as pd
    import numpy as np
    from faker import Faker
    import random

    print(">>> [3] Тяжелые библиотеки (Pandas/Faker) загружены")
except ImportError as e:
    print(f"!!! ОШИБКА ИМПОРТА: {e}")
    exit()

fake = Faker()
Faker.seed(42)
np.random.seed(42)


def generate_mobile_banking_data(num_rows=20000):
    print(f">>> [4] Начинаю генерацию {num_rows} строк...")

    data = []
    clients = [fake.uuid4() for _ in range(int(num_rows / 10))]

    for i in range(num_rows):
        if i % 5000 == 0:
            print(f"    ... обработано {i} транзакций")

        client = random.choice(clients)
        is_fraud = 0

        amount = round(np.random.uniform(500, 500000), 2)
        hour = np.random.randint(0, 24)
        is_night = 1 if 0 <= hour <= 5 else 0
        device_os = random.choice(['Android', 'iOS', 'Web'])
        is_new_device = random.choice([0, 1])

        if is_new_device and is_night and amount > 20000 and random.random() > 0.15:
            is_fraud = 1
        elif amount > 150000 and random.random() > 0.4:
            is_fraud = 1
        elif random.random() < 0.01:
            is_fraud = 1

        data.append({
            'transaction_id': fake.uuid4(),
            'client_id': client,
            'amount_kzt': amount,
            'transaction_hour': hour,
            'is_night': is_night,
            'device_os': device_os,
            'is_new_device': is_new_device,
            'time_on_screen_sec': np.random.randint(5, 300),
            'merchant_category': random.choice(['P2P', 'Grocery']),
            'is_fraud': is_fraud
        })

    df = pd.DataFrame(data)
    print(f">>> [5] Генерация завершена. Fraud Rate: {df['is_fraud'].mean():.2%}")
    return df


if __name__ == "__main__":
    print(">>> [Main] Запуск функции main")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '..', 'data')

    print(f">>> [Main] Папка для сохранения: {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(">>> [Main] Папка создана")

    output_path = os.path.join(output_dir, 'transactions_train.csv')

    df = generate_mobile_banking_data(20000)

    try:
        df.to_csv(output_path, index=False)
        print(f">>> [УСПЕХ] Файл сохранен: {output_path}")
    except Exception as e:
        print(f"!!! ОШИБКА СОХРАНЕНИЯ: {e}")