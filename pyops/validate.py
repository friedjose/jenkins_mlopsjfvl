import pandas as pd
import os

def validate_datasets():
    print("🚀 Iniciando validaciones de datasets...")

    base_path = os.path.join(os.getcwd(), "repo_modelado_JoseVL")
    files = [
        "dataset_no_scaling.csv",
        "dataset_scaling.csv"
    ]

    for file in files:
        path = os.path.join(base_path, file)
        if not os.path.exists(path):
            print(f"❌ Archivo no encontrado: {file}")
            continue

        df = pd.read_csv(path)
        print(f"✅ {file} cargado correctamente con {df.shape[0]} filas y {df.shape[1]} columnas.")

        null_count = df.isnull().sum().sum()
        if null_count > 0:
            print(f"⚠️ {file} contiene {null_count} valores nulos.")
        else:
            print(f"✅ {file} no contiene valores nulos.")

if __name__ == "__main__":
    validate_datasets()
