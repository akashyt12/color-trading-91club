from flask import Flask, render_template, jsonify
import requests, json, os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ================= CONFIG =================
CSV_PATH = "notex_training_data.csv"
MODEL_PATH = "notex_bigsmall_model_v4.joblib"
SCALER_PATH = "notex_scaler_v4.joblib"
API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json?pageSize=4&pageNo=1"

app = Flask(__name__)

# ====== MODEL SETUP ======
def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    df = pd.read_csv(CSV_PATH)
    for c in ['num1','num2','num3','num4','next']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['sum_last3'] = df[['num2','num3','num4']].sum(axis=1)
    df['change'] = df['num4'] - df['num3']
    X = df[['num1','num2','num3','num4','sum_last3','change']].values
    y = (df['next'] >= 5).astype(int).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(Xs, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler

model, scaler = load_or_train_model()

def fetch_last4():
    try:
        r = requests.get(API_URL, timeout=8)
        data = r.json()
        lst = data.get('data', {}).get('list', [])
        nums = []
        for item in lst[:4]:
            nums.append(int(item.get('number', 0)))
        if len(nums) == 4:
            return list(reversed(nums))
    except:
        return []
    return []

def make_features_live(nums):
    n1, n2, n3, n4 = nums
    sum_last3 = n2+n3+n4
    change = n4-n3
    return np.array([[n1,n2,n3,n4,sum_last3,change]])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_live():
    nums = fetch_last4()
    if len(nums) != 4:
        return jsonify({'error': 'Server Connect Failed ❌'})
    X_live = make_features_live(nums)
    Xs = scaler.transform(X_live)
    probas = model.predict_proba(Xs)[0]
    pred = int(np.argmax(probas))
    result = "BIG 🔴" if pred == 1 else "SMALL 🟢"
    return jsonify({
        'nums': nums,
        'prediction': result,
        'confidence': f"{probas[pred]*100:.2f}%",
        'time': datetime.now().strftime("%H:%M:%S"),
        'status': "SERVER CONNECTED ✅"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
