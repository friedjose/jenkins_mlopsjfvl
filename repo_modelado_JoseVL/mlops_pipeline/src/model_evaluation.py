# model_evaluation.py
import requests
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 1. Configuración
# ============================
API_URL = "http://localhost:8000/predict_batch"  # endpoint FastAPI
DATASET_PATH = "C:/Users/jose5/Proyecto-MLops/dataset_scaling.csv"
TARGET = "Pago_atiempo"

# ============================
# 2. Cargar datos
# ============================
st.title("📊 Evaluación del Modelo Desplegado")

st.write("Cargando dataset escalado...")
df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# mismo split que en ft_engineering
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.write("✅ Dataset cargado con éxito")
st.write(f"Tamaño X_test: {X_test.shape}, y_test: {y_test.shape}")

# ============================
# 3. Llamar al endpoint
# ============================
st.write("Obteniendo predicciones del modelo vía API...")

payload = {"batch": X_test.values.tolist()}
response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    preds = response.json()["predictions"]
else:
    st.error("❌ Error al conectar con la API.")
    st.stop()

# ============================
# 4. Métricas
# ============================
st.subheader("Métricas de Clasificación")

report = classification_report(y_test, preds, output_dict=True, zero_division=0)
st.json(report)

# ============================
# 5. Matriz de confusión
# ============================
st.subheader("Matriz de Confusión")
cm = confusion_matrix(y_test, preds)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1], ax=ax)
ax.set_xlabel("Predicciones")
ax.set_ylabel("Reales")
st.pyplot(fig)

# ============================
# 6. Curva ROC
# ============================
st.subheader("Curva ROC")
# como FastAPI devuelve probabilidades, podemos calcular la ROC
if "probabilities" in response.json():
    probs = response.json()["probabilities"]
    probs_clase1 = [p[1] for p in probs]
    fpr, tpr, _ = roc_curve(y_test, probs_clase1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1], [0,1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    st.pyplot(fig)
else:
    st.warning("El modelo no retornó probabilidades, no se puede calcular ROC.")
