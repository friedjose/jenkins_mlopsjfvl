import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


# =====================================================
# 1. Limpieza de datos
# =====================================================
def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza inicial: eliminar columnas redundantes, imputar nulos y ajustar rangos."""
    
    # Eliminar variables con alto % de nulos
    df = df.drop(columns=["promedio_ingresos_datacredito", "tendencia_ingresos"], errors="ignore")

    # Imputar numéricas con mediana
    numeric_vars = ['saldo_mora', 'saldo_total', 'saldo_principal', 'puntaje_datacredito']
    for var in numeric_vars:
        if var in df.columns:
            df[var].fillna(df[var].median(), inplace=True)

    # Validar rango de edad
    edad_min, edad_max = 18, 100
    df.loc[(df["edad_cliente"] < edad_min) | (df["edad_cliente"] > edad_max), "edad_cliente"] = np.nan
    df["edad_cliente"].fillna(df["edad_cliente"].median(), inplace=True)

    # Eliminar columnas redundantes detectadas
    df = df.drop(columns=["saldo_mora_codeudor", "saldo_total", "puntaje", "creditos_sectorFinanciero"], errors="ignore")

    return df


# =====================================================
# 2. Crear pipelines
# =====================================================
def crear_pipelines(binarias, politomicas, numeric_features):
    """Genera dos pipelines: con y sin escalado"""
    
    # Sin escalado (heurístico)
    preprocessor_no_scaling = ColumnTransformer(
        transformers=[
            ("bin", OneHotEncoder(drop="first", handle_unknown="ignore"), binarias),
            ("poly", OneHotEncoder(drop=None, handle_unknown="ignore"), politomicas),
            ("num", "passthrough", numeric_features)
        ]
    )
    pipeline_no_scaling = Pipeline([("preprocessor", preprocessor_no_scaling)])

    # Con escalado (modelos ML)
    preprocessor_scaling = ColumnTransformer(
        transformers=[
            ("bin", OneHotEncoder(drop="first", handle_unknown="ignore"), binarias),
            ("poly", OneHotEncoder(drop=None, handle_unknown="ignore"), politomicas),
            ("num", MinMaxScaler(), numeric_features)
        ]
    )
    pipeline_scaling = Pipeline([("preprocessor", preprocessor_scaling)])

    return pipeline_no_scaling, pipeline_scaling


# =====================================================
# 3. Transformar datos
# =====================================================
def transformar_datos(X, y, pipeline, nombre_salida: str):
    """Aplica pipeline, devuelve DataFrame procesado y lo guarda."""
    
    X_transformed = pipeline.fit_transform(X)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    X_df = pd.DataFrame(
        X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed,
        columns=feature_names
    )
    df_final = pd.concat([X_df, y.reset_index(drop=True)], axis=1)

    # Exportar
    df_final.to_csv(f"{nombre_salida}.csv", index=False)
    df_final.to_parquet(f"{nombre_salida}.parquet", index=False)

    return df_final


# =====================================================
# 4. Split train/test
# =====================================================
def split_datos(df, target="Pago_atiempo", test_size=0.2, random_state=42):
    """Divide el dataset en entrenamiento y prueba"""
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# =====================================================
# 5. Main
# =====================================================
def main():
    # Cargar dataset
    df = pd.read_csv("mlops_pipeline\df_post_eda.csv")

    # Limpieza
    df = limpiar_datos(df)

    # Definir variables
    X = df.drop(columns=["Pago_atiempo"])
    y = df["Pago_atiempo"]
    binarias = ["tipo_laboral"]
    politomicas = ["tipo_credito"]
    numeric_features = [col for col in X.columns if col not in binarias + politomicas]

    # Crear pipelines
    pipe_no_scaling, pipe_scaling = crear_pipelines(binarias, politomicas, numeric_features)

    # Transformar y exportar
    df_no_scaling = transformar_datos(X, y, pipe_no_scaling, "dataset_no_scaling")
    df_scaling = transformar_datos(X, y, pipe_scaling, "dataset_scaling")

    # Dividir train/test
    X_train, X_test, y_train, y_test = split_datos(df_scaling)

    print("✅ Dataset sin escalar:", df_no_scaling.shape)
    print("✅ Dataset con escalado:", df_scaling.shape)
    print("✅ Train shape:", X_train.shape, "Test shape:", X_test.shape)


if __name__ == "__main__":
    main()