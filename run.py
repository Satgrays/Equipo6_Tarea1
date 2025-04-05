import subprocess
import sys
import os

def install_requirements():
    req_path = "requirements.txt"
    print("🔧 Instalando dependencias")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
    print("✅ Requerimientos instalados.")

def run_script(script_name):
    script_path = os.path.join("scripts", script_name)
    print(f"🚀 Ejecutando {script_name}...")
    subprocess.check_call([sys.executable, script_path])
    print(f"✅ {script_name} completado.\n")

if __name__ == "__main__":
    try:
        install_requirements()
        run_script("Data.py")    # Procesamiento de datos
        run_script("Model.py")   # Entrenamiento del modelo
        print("🎉 Ejecución completa de run.py.")
    except subprocess.CalledProcessError as e:
        print("❌ Error durante la ejecución de un script:", e)
        sys.exit(1)
