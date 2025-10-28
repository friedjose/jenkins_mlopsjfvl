import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

# ============================
# 1. Funciones auxiliares
# ============================

def summarize_classification(y_true, y_pred):
    """Resumen de métricas de clasificación (globales y por clase)"""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        # Métricas por clase
        "recall_clase_0 (morosos)": report["0"]["recall"],
        "recall_clase_1 (buenos pagadores)": report["1"]["recall"],
        "precision_clase_0 (morosos)": report["0"]["precision"],
        "precision_clase_1 (buenos pagadores)": report["1"]["precision"],
        "f1_clase_0 (morosos)": report["0"]["f1-score"],
        "f1_clase_1 (buenos pagadores)": report["1"]["f1-score"],
    }


def build_model(classifier_fn, data_params: dict, test_frac: float = 0.2, use_smote: bool = False):
    """Entrenar un modelo de clasificación y evaluar en train y test"""
    # Extraer parámetros
    name_of_y_col = data_params["name_of_y_col"]
    names_of_x_cols = data_params["names_of_x_cols"]
    dataset = data_params["dataset"]

    # Dividir en features y target
    X = dataset[names_of_x_cols]
    Y = dataset[name_of_y_col]

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_frac, random_state=1234, stratify=Y
    )

    # Pipeline con o sin SMOTE
    if use_smote:
        classifier_pipe = ImbPipeline(steps=[
            ("smote", SMOTE(random_state=1234)),
            ("model", classifier_fn)
        ])
    else:
        classifier_pipe = Pipeline(steps=[("model", classifier_fn)])

    # Entrenamiento
    model = classifier_pipe.fit(x_train, y_train)

    # Predicciones
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    # Resúmenes
    return model, {
        "train": summarize_classification(y_train, y_pred_train),
        "test": summarize_classification(y_test, y_pred_test),
    }


# ============================
# 2. Main
# ============================

if __name__ == "__main__":
    # Cargar dataset escalado
    df_scaling = pd.read_csv("C:/Users/jose5/proyecto-mlops/dataset_scaling.csv")

    # Definir columnas
    target = "Pago_atiempo"
    features_scaling = [col for col in df_scaling.columns if col != target]

    # Modelos a probar
    models = {
        #"logistic": LogisticRegression(max_iter=1000, solver="liblinear"),
        "svc": SVC(C=1.0, probability=True),
        "naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(),
        "bagging": BaggingClassifier(),
        "sgd": SGDClassifier(max_iter=1000),
        "xgboost": XGBClassifier(eval_metric="logloss", random_state=1234),
        "balanced_rf": BalancedRandomForestClassifier(random_state=1234),
    }

    # Entrenar y evaluar
    results = {}
    trained_models = {}

    print("\n=== Entrenando modelos ===")
    for model_name, model in models.items():
        print(f"\nModelo: {model_name}")
        trained_model, metrics = build_model(model, {
            "name_of_y_col": target,
            "names_of_x_cols": features_scaling,
            "dataset": df_scaling
        }, use_smote=True)  # <- aquí puedes alternar con False para probar sin SMOTE
        results[model_name] = metrics
        trained_models[model_name] = trained_model

    # Crear tabla resumen
    resumen = []
    for name, metrics in results.items():
        test = metrics["test"]
        resumen.append({
            "Modelo": name,
            "Accuracy": test["accuracy"],
            "Recall_morosos": test["recall_clase_0 (morosos)"],
            "Precision_morosos": test["precision_clase_0 (morosos)"],
            "F1_morosos": test["f1_clase_0 (morosos)"],
            "F1_weighted": test["f1_weighted"],
        })

    df_resumen = pd.DataFrame(resumen)
    print("\n============================================================")
    print("TABLA RESUMEN DE MODELOS (Test set)")
    print("============================================================")
    print(df_resumen.to_string(index=False))

    # Gráficos comparativos
    df_resumen.set_index("Modelo")[["Recall_morosos", "Precision_morosos", "F1_morosos"]].plot.bar(figsize=(12,6))
    plt.title("Comparación de métricas para clase morosos (Test)")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    df_resumen.set_index("Modelo")[["Accuracy", "F1_weighted"]].plot.bar(figsize=(12,6))
    plt.title("Métricas generales (Test)")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Seleccionar mejor modelo (priorizando F1 de morosos)
    best_model_name = df_resumen.sort_values(by="F1_morosos", ascending=False).iloc[0]["Modelo"]
    best_model = trained_models[best_model_name]

    print("\n============================================================")
    print(f"🏆 Mejor modelo seleccionado: {best_model_name}")
    print("============================================================")

    # Guardar modelo
    joblib.dump(best_model, "best_model.pkl")
    print("✅ Modelo guardado en 'best_model.pkl'")
