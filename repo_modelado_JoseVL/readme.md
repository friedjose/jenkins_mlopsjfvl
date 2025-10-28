# 🚀 MLOps Pipeline - Predicción de Riesgo Crediticio

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker)](https://www.docker.com/)

Pipeline completo de MLOps para predicción de pago puntual de créditos, incluyendo feature engineering, entrenamiento de modelos, despliegue via API, evaluación y monitoreo en producción.

## 🎯 Características Principales

- **Feature Engineering**: Limpieza, encoding categórico y escalado de datos
- **Multi-model Training**: Comparación automática de 8 algoritmos ML
- **REST API**: Predicciones individuales y en lote con FastAPI
- **Dashboard de Evaluación**: Métricas visuales con Streamlit
- **Monitoreo Continuo**: Detección de data drift con EvidentlyAI
- **Containerización**: Despliegue con Docker

## 📁 Estructura del Proyecto

```
mlops_pipeline/src/
├── cargar_datos.ipynb         # Notebook carga inicial de datos
├── comprension_eda.ipynb     # Análisis exploratorio de datos
├── config.json               # Configuración del pipeline
├── ft_engineering.py         # Feature engineering
├── heuristic_model.py        # Modelo baseline heurístico
├── model_training.py         # Entrenamiento multi-modelo
├── model_deploy.py           # API REST con FastAPI
├── model_evaluation.py       # Dashboard de métricas
├── model_monitoring.py       # Monitoreo y drift detection
└── monitoring_log.csv        # Logs de predicciones
```

## ⚡ Quick Start

### 0. Análisis Exploratorio (Opcional)
```bash
# Revisar notebooks para entender los datos
jupyter notebook mlops_pipeline/src/cargar_datos.ipynb
jupyter notebook mlops_pipeline/src/comprension_eda.ipynb
```

### 1. Instalación
```bash
pip install -r requirements.txt
```

### 2. Pipeline Completo
```bash
# Feature engineering
python mlops_pipeline/src/ft_engineering.py

# Entrenamiento (selecciona automáticamente el mejor modelo)
python mlops_pipeline/src/model_training.py

# Despliegue de API
docker build -t credit-mlops .
docker run -p 8000:8000 credit-mlops
```

### 3. Dashboards
```bash
# Evaluación de métricas
streamlit run mlops_pipeline/src/model_evaluation.py

# Monitoreo en producción
streamlit run mlops_pipeline/src/model_monitoring.py
```

## 🔌 API Endpoints

- **Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **POST** `/predict_one` - Predicción individual
- **POST** `/predict_batch` - Predicción en lote

## 🤖 Modelos Soportados

El sistema entrena y compara automáticamente:
- SVM, Naive Bayes, Decision Tree
- Random Forest, Bagging, SGD
- XGBoost, Balanced Random Forest

**Selección**: Mejor F1-score para clase minoritaria (morosos)

## 📊 Monitoreo

- **Logs automáticos** de predicciones en `monitoring_log.csv`
- **Data drift detection** con reportes visuales
- **Métricas en tiempo real** via dashboard Streamlit

## 🛠️ Stack Tecnológico

- **ML**: scikit-learn, XGBoost, imbalanced-learn
- **API**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **Monitoreo**: EvidentlyAI
- **Deployment**: Docker

## 📈 Resultados

- API REST lista para producción
- Dashboard interactivo de métricas
- Sistema de alertas por data drift
- Pipeline automatizado end-to-end
## 👨‍💻 Autor
Jose Fernando Villegas Lora 

## 👨‍💻 Supervisora
Clara Isabel Otalvaro Agudelo