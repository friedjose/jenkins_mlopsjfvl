
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


# =====================================================
# 1. Definir modelo heurístico
# =====================================================
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class HeuristicModel(BaseEstimator, ClassifierMixin):
    def __init__(self, puntaje_threshold=740, huella_threshold=8, mora_threshold=0,
                 ingresos_threshold=5_000_000, creditos_threshold=10):
        self.puntaje_threshold = puntaje_threshold
        self.huella_threshold = huella_threshold
        self.mora_threshold = mora_threshold
        self.ingresos_threshold = ingresos_threshold
        self.creditos_threshold = creditos_threshold

    def fit(self, X, y=None):
        if y is not None:
            self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        predictions = []

        for _, row in X.iterrows():
            # Regla 1: Puntaje muy bajo → riesgo
            if row["num__puntaje_datacredito"] < self.puntaje_threshold:
                predictions.append(0)

            # Regla 2: Muchas consultas → riesgo
            elif row["num__huella_consulta"] > self.huella_threshold:
                predictions.append(0)

            # Regla 3: Mora activa → riesgo
            elif row["num__saldo_mora"] > self.mora_threshold:
                predictions.append(0)

            # Regla 4: Tipo de crédito 6 → riesgo, salvo buen perfil
            elif ("poly__tipo_credito_6" in X.columns and row["poly__tipo_credito_6"] == 1 and
                  not (row["num__salario_cliente"] > self.ingresos_threshold and 
                       row["num__puntaje_datacredito"] >= self.puntaje_threshold)):
                predictions.append(0)

            # Regla 5: Muchos créditos vigentes → riesgo
            elif row["num__cant_creditosvigentes"] > self.creditos_threshold:
                predictions.append(0)

            # Regla 6: Buenos ingresos + buen puntaje → paga a tiempo
            elif (row["num__salario_cliente"] > self.ingresos_threshold and 
                  row["num__puntaje_datacredito"] >= self.puntaje_threshold):
                predictions.append(1)

            # Caso por defecto: asumimos buen comportamiento
            else:
                predictions.append(1)

        return np.array(predictions)


# =====================================================
# 2. Cargar dataset
# =====================================================
df = pd.read_csv("C:/Users/jose5/proyecto-prueba/dataset_no_scaling1.csv")

X = df.drop(columns=["Pago_atiempo"])
y = df["Pago_atiempo"]

# Split train/test
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =====================================================
# 3. Evaluación con validación cruzada
# =====================================================
scoring_metrics = ["accuracy", "f1", "precision", "recall"]
kfold = KFold(n_splits=10)

model = HeuristicModel()
model_pipe = Pipeline(steps=[("model", model)])

cv_results = {}
train_results = {}

for metric in scoring_metrics:
    cv_results[metric] = cross_val_score(model_pipe, x_train, y_train, cv=kfold, scoring=metric)
    model_pipe.fit(x_train, y_train)
    train_results[metric] = model_pipe.score(x_train, y_train)

cv_results_df = pd.DataFrame(cv_results)

# =====================================================
# 4. Resultados
# =====================================================
print("\nResultados validación cruzada:")
print(cv_results_df.mean())

print("\nResultados en train:")
print(train_results)

# Boxplot CV
cv_results_df.plot.box(title="Cross Validation Boxplot", ylabel="Score")
plt.show()

# Comparación train vs CV
means = cv_results_df.mean()
stds = cv_results_df.std()

plt.figure(figsize=(8, 6))
x_pos = range(len(scoring_metrics))
plt.bar(x_pos, [train_results[m] for m in scoring_metrics], width=0.4, label="Train Score")
plt.bar(
    [i + 0.4 for i in x_pos],
    means,
    width=0.4,
    yerr=stds,
    capsize=5,
    label="CV Mean",
)
plt.xticks([i + 0.2 for i in x_pos], scoring_metrics)
plt.ylabel("Score")
plt.title("Training vs Cross-Validation Scores")
plt.legend()
plt.show()

# =====================================================
# 5. Matriz de confusión
# =====================================================
y_pred = model_pipe.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="viridis", values_format="d")
plt.show()

# =====================================================
# 6. Curva de aprendizaje
# =====================================================
train_sizes, train_scores, test_scores = learning_curve(
    model_pipe, x_train, y_train, cv=kfold, scoring="recall"
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training score")
plt.plot(train_sizes, test_scores.mean(axis=1), "o-", label="Cross-validation score")
plt.title("Learning Curve for HeuristicModel")
plt.xlabel("Training examples")
plt.ylabel("Recall")
plt.legend()
plt.show()
