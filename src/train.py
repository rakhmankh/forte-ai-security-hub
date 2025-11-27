print(">>> [1] Инициализация скрипта обучения...")

import os
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap
import joblib


def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'transactions_train.csv')
    models_dir = os.path.join(current_dir, '..', 'models')

    print(f">>> [2] Ищу данные здесь: {data_path}")

    if not os.path.exists(data_path):
        print("!!! ОШИБКА: Файл с данными не найден.")
        return

    print(">>> [3] Читаю CSV файл...")
    df = pd.read_csv(data_path)
    print(f"    Загружено строк: {len(df)}")

    drop_cols = ['is_fraud', 'transaction_id']

    if 'client_id' in df.columns:
        drop_cols.append('client_id')

    X = df.drop(drop_cols, axis=1)
    y = df['is_fraud']

    print(f">>> Признаки для обучения: {list(X.columns)}")

    cat_features = ['device_os', 'merchant_category']

    print(">>> [4] Разделение на Train/Test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(">>> [5] Старт обучения модели (CatBoost)...")
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_features,
        auto_class_weights='Balanced',
        verbose=50
    )

    model.fit(X_train, y_train)

    print("\n>>> [6] Оценка качества (Performance):")
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))
    try:
        roc_auc = roc_auc_score(y_test, preds_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    except:
        print("ROC-AUC не посчитался (возможно, в тесте только один класс)")

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_save_path = os.path.join(models_dir, 'fraud_model.cbm')
    explainer_save_path = os.path.join(models_dir, 'shap_explainer.joblib')

    print(f">>> [7] Сохранение модели в: {model_save_path}")
    model.save_model(model_save_path)

    print(">>> [8] Генерация SHAP Explainer...")
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, explainer_save_path)

    print(">>> [УСПЕХ] Модель переобучена на РЕАЛЬНЫХ данных!")


if __name__ == "__main__":
    train()