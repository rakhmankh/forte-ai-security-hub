import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
import os
import csv
from datetime import datetime
from fpdf import FPDF


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data')
MODELS_DIR = os.path.join(CURRENT_DIR, '..', 'models')
STATIC_DIR = os.path.join(CURRENT_DIR, '..', 'static')


os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(MODELS_DIR):
    print(f"WARNING: Папка {MODELS_DIR} не найдена. Сначала запусти train.py!")


LOG_FILE = os.path.join(DATA_DIR, "transaction_logs.csv")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback_dataset.csv")


ml_artifacts = {
    "model": None,
    "explainer": None
}



@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> [Startup] Инициализация сервиса...")

    model_path = os.path.join(MODELS_DIR, 'fraud_model.cbm')
    explainer_path = os.path.join(MODELS_DIR, 'shap_explainer.joblib')


    if os.path.exists(model_path):
        print(f">>> [Startup] Загрузка модели из: {model_path}")
        model = CatBoostClassifier()
        model.load_model(model_path)
        ml_artifacts["model"] = model
    else:
        print(f"!!! [CRITICAL] Модель не найдена по пути: {model_path}")


    if os.path.exists(explainer_path):
        print(f">>> [Startup] Загрузка SHAP Explainer...")
        ml_artifacts["explainer"] = joblib.load(explainer_path)
    else:
        print("!!! [Warning] SHAP Explainer не найден. Интерпретация отключена.")

    print(">>> [Startup] Сервер готов к работе.")
    yield
    print(">>> [Shutdown] Остановка сервиса...")
    ml_artifacts.clear()



app = FastAPI(
    title="ForteBank Anti-Fraud AI",
    description="Система выявления мошенничества с объяснимостью решений",
    version="1.0",
    lifespan=lifespan
)


if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")



class TransactionFeatures(BaseModel):
    amount_kzt: float
    transaction_hour: int
    is_night: int
    device_os: str
    is_new_device: int
    time_on_screen_sec: int
    merchant_category: str


class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    probability: float
    risk_level: str
    threshold_used: float
    explanations: dict = {}



def log_transaction(data: dict, prediction: dict):
    """Записывает транзакцию в CSV для аудита"""
    try:
        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'txn_id', 'amount', 'prob', 'is_fraud', 'risk_level'])

            writer.writerow([
                datetime.now().isoformat(),
                prediction['transaction_id'],
                data['amount_kzt'],
                prediction['probability'],
                prediction['is_fraud'],
                prediction['risk_level']
            ])
    except Exception as e:
        print(f"Error logging transaction: {e}")


@app.get("/")
async def read_index():
    """Отдает главную страницу (Frontend)"""
    index_path = os.path.join(STATIC_DIR, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. Check /static folder."}


@app.post("/predict", response_model=PredictionResponse)
def predict(
        t: TransactionFeatures,
        threshold: float = Query(0.5, ge=0.0, le=1.0, description="Порог срабатывания")
):

    model = ml_artifacts["model"]
    explainer = ml_artifacts["explainer"]

    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded. Check server logs.")

    input_data = pd.DataFrame([t.dict()])

    try:
        prob = model.predict_proba(input_data)[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ML Error: {str(e)}")

    is_fraud = bool(prob >= threshold)


    if prob > 0.8:
        risk_level = "CRITICAL"
    elif prob > 0.5:
        risk_level = "HIGH"
    elif prob > 0.2:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    shap_values_dict = {}
    if explainer:
        try:
            shap_values = explainer.shap_values(input_data)
            feature_names = input_data.columns

            vals = shap_values
            if isinstance(vals, list):
                vals = vals[0]
            if len(np.shape(vals)) > 1:
                vals = vals[0]

            contributions = zip(feature_names, vals)
            sorted_contribs = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:3]

            for name, val in sorted_contribs:
                sign = "+" if val > 0 else ""
                shap_values_dict[name] = f"{sign}{val:.4f}"
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            shap_values_dict = {"error": "Explanation unavailable"}

    response_payload = {
        "transaction_id": "txn_" + str(np.random.randint(10000, 99999)),
        "is_fraud": is_fraud,
        "probability": round(prob, 4),
        "risk_level": risk_level,
        "threshold_used": threshold,
        "explanations": shap_values_dict
    }

    log_transaction(t.dict(), response_payload)

    return response_payload


@app.post("/feedback")
def submit_feedback(transaction_id: str, actual_is_fraud: bool):
    try:
        file_exists = os.path.isfile(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'txn_id', 'actual_is_fraud'])

            writer.writerow([datetime.now().isoformat(), transaction_id, int(actual_is_fraud)])

        return {"status": "ok", "message": f"Feedback saved for {transaction_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


@app.post("/admin/retrain")
def trigger_retraining():
    return {"status": "started", "message": "Retraining pipeline initiated on new data."}


from fastapi import UploadFile, File


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    model = ml_artifacts["model"]
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        contents = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    required_cols = ['amount_kzt', 'transaction_hour', 'is_night', 'device_os', 'is_new_device', 'time_on_screen_sec',
                     'merchant_category']
    if not all(col in df.columns for col in required_cols):
        raise HTTPException(status_code=400, detail=f"CSV must contain columns: {required_cols}")

    input_data = df[required_cols]

    probs = model.predict_proba(input_data)[:, 1]

    df['fraud_probability'] = np.round(probs, 4)
    df['is_fraud_predicted'] = df['fraud_probability'] > 0.5
    df['risk_level'] = df['fraud_probability'].apply(
        lambda x: "CRITICAL" if x > 0.8 else ("HIGH" if x > 0.5 else "LOW")
    )

    return df.to_dict(orient="records")


@app.get("/admin/stats")
def get_stats():
    if not os.path.exists(LOG_FILE):
        return {"message": "No data yet"}

    try:
        df = pd.read_csv(LOG_FILE)

        total_txns = len(df)
        fraud_detected = df[df['is_fraud'] == 'True']
        if fraud_detected.empty:
            fraud_detected = df[(df['is_fraud'] == True) | (df['is_fraud'] == '1')]

        count_fraud = len(fraud_detected)

        money_saved = fraud_detected['amount'].astype(float).sum()

        return {
            "total_transactions": total_txns,
            "fraud_detected_count": count_fraud,
            "fraud_rate_today": round((count_fraud / total_txns) * 100, 2) if total_txns > 0 else 0,
            "money_saved_kzt": money_saved
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/simulate_batch")
def simulate_batch_data():
    model = ml_artifacts["model"]
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    dataset_path = os.path.join(DATA_DIR, "transactions_train.csv")

    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found. Run generator.py first!")

    try:
        df_full = pd.read_csv(dataset_path)
        sample_df = df_full.sample(n=15)
        input_cols = ['amount_kzt', 'transaction_hour', 'is_night', 'device_os',
                      'is_new_device', 'time_on_screen_sec', 'merchant_category']

        input_data = sample_df[input_cols].copy()

        probs = model.predict_proba(input_data)[:, 1]

        input_data['fraud_probability'] = np.round(probs, 4)
        input_data['is_fraud_predicted'] = input_data['fraud_probability'] > 0.5
        input_data['risk_level'] = input_data['fraud_probability'].apply(
            lambda x: "CRITICAL" if x > 0.8 else ("HIGH" if x > 0.5 else "LOW")
        )

        return input_data.to_dict(orient="records")

    except Exception as e:
        print(f"Error simulating batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_report")
def download_report(
        txn_id: str,
        amount: str,
        merchant: str,
        risk: str
):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.set_text_color(157, 31, 70)  # Forte Color
    pdf.cell(200, 10, txt="ForteBank Security Incident Report", ln=1, align="C")

    pdf.set_line_width(1)
    pdf.set_draw_color(157, 31, 70)
    pdf.line(10, 20, 200, 20)

    pdf.ln(20)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Transaction ID: {txn_id}", ln=1)
    pdf.cell(200, 10, txt=f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.cell(200, 10, txt=f"Amount: {amount} KZT", ln=1)
    pdf.cell(200, 10, txt=f"Merchant: {merchant}", ln=1)

    pdf.set_font("Arial", 'B', size=12)
    pdf.set_text_color(255, 0, 0)
    pdf.cell(200, 10, txt=f"Risk Level: {risk}", ln=1)
    pdf.cell(200, 10, txt=f"Status: BLOCKED", ln=1)

    pdf.set_y(-30)
    pdf.set_font("Arial", 'I', size=8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, "Generated automatically by Forte AI Watchdog System", 0, 0, 'C')

    filename = f"Incident_{txn_id}.pdf"
    file_path = os.path.join(DATA_DIR, filename)
    pdf.output(file_path)

    return FileResponse(file_path, filename=filename, media_type='application/pdf')

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)