# config.py
"""
Módulo de configuración central para el proyecto U-Net.
Contiene rutas, hiperparámetros y constantes de ejecución.
"""
import os

# --- Rutas del Sistema ---
# Modifica esto según tu entorno (local o servidor)
BASE_DIR = "/media/ATADISK/njorge/CM6New/"
CHECKPOINT_DIR = "./M_S2UResidual_IPSL_GFDL_CNRM"
FINAL_MODEL_PATH = os.path.join(BASE_DIR, "S2U_Res_IPSL_GFDL_CNRM.keras")

# --- Hiperparámetros ---
SALTO = 16
SEED = 42
EPOCHS = 1000
BATCH_SIZE = 16 # Ajusta según tu memoria GPU/CPU
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-3