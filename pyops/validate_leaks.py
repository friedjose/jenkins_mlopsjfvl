import os
import re
import sys

# --- Expresiones regulares para detectar posibles secretos ---
patterns = {
    "AWS Secret": re.compile(r"AKIA[0-9A-Z]{16}"),
    "Private Key": re.compile(r"-----BEGIN (RSA|DSA|EC|PGP) PRIVATE KEY-----"),
    "Password": re.compile(r"password\s*=\s*['\"]?.+['\"]?", re.IGNORECASE),
    "Token": re.compile(r"token\s*=\s*['\"]?[A-Za-z0-9_\-]{10,}['\"]?", re.IGNORECASE),
    "API Key": re.compile(r"api[_-]?key\s*=\s*['\"]?[A-Za-z0-9_\-]{10,}['\"]?", re.IGNORECASE),
    "Authorization Header": re.compile(r"Authorization:\s*Bearer\s+[A-Za-z0-9\-_=\.]+", re.IGNORECASE),
}

# --- Archivos o carpetas que no se revisarán ---
exclude_dirs = {'.git', 'venv', '__pycache__'}

print("🔍 Escaneando repositorio en busca de credenciales filtradas...\n")
found_leaks = []

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    for file in files:
        path = os.path.join(root, file)
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for name, pattern in patterns.items():
                    if pattern.search(content):
                        found_leaks.append((path, name))
        except Exception as e:
            print(f"⚠️ No se pudo leer {path}: {e}")

# --- Resultados ---
if found_leaks:
    print("❌ Posibles credenciales encontradas:")
    for path, name in found_leaks:
        print(f"   - [{name}] en {path}")

    print("\n🚨 ERROR: Se detectaron posibles llaves o credenciales filtradas. Corrige antes de continuar.")
    sys.exit(1)  # Falla el pipeline
else:
    print("✅ No se encontraron credenciales ni llaves expuestas.")
    sys.exit(0)
