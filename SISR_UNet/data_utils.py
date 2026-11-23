# data_utils.py
"""
Utilidades para la gestión de memoria y carga del dataset climático.
"""
import os
import gc
import logging
import numpy as np
import tensorflow as tf
from config import BASE_DIR, SALTO

logger = logging.getLogger(__name__)

def clear_memory() -> None:
    """Libera la memoria RAM y reinicia la sesión de Keras."""
    gc.collect()
    tf.keras.backend.clear_session()
    logger.info("Memoria RAM y sesión de Keras liberadas.")

def normalize(data: np.ndarray) -> np.ndarray:
    """Normaliza un arreglo usando la media y desviación estándar (ignorando NaNs)."""
    return (data - np.nanmean(data)) / np.nanstd(data)

def load_dataset() -> tuple:
    """
    Carga, normaliza y concatena los datos de temperatura y vientos (IPSL, CNRM, GFDL).
    
    Returns:
        tuple: (DataT, DataU, DataV, Label, M, T)
    """
    logger.info("Iniciando carga de datos...")
    
    # --- Variables T (Temperatura) ---
    T_IPSL = normalize(np.load(os.path.join(BASE_DIR, "TOS_LR_IPSL_Inter1994_2100.npy"))[8:41, 3:56,:])
    T_CNRM = normalize(np.load(os.path.join(BASE_DIR, "TOS_LR_CNRM_Inter1994_2100.npy"))[8:41, 3:56,:])
    T_GFDL = normalize(np.load(os.path.join(BASE_DIR, "TOS_LR_GFDL_Inter1994_2100.npy"))[8:41, 3:56,:])
    
    DataTi = np.concatenate((T_IPSL[:, :, ::SALTO], T_CNRM[:, :, ::SALTO], T_GFDL[:, :, ::SALTO]), axis=2)
    T = DataTi.shape[2]
    DataT = np.transpose(DataTi, (2, 0, 1))[:, :, :, np.newaxis]
    del T_IPSL, T_CNRM, T_GFDL, DataTi; gc.collect()

    # --- Variables U (Viento Zonal) ---
    U_IPSL = normalize(np.load(os.path.join(BASE_DIR, "UAS_LR_IPSL_Inter1994_2100.npy")))
    U_CNRM = normalize(np.load(os.path.join(BASE_DIR, "UAS_LR_CNRM_Inter1994_2100.npy")))
    U_GFDL = normalize(np.load(os.path.join(BASE_DIR, "UAS_LR_GFDL_Inter1994_2100.npy")))
    
    DataUi = np.concatenate((U_IPSL[:, :, ::SALTO], U_CNRM[:, :, ::SALTO], U_GFDL[:, :, ::SALTO]), axis=2)
    DataU = np.transpose(DataUi, (2, 0, 1))[:, :, :, np.newaxis]
    del U_IPSL, U_CNRM, U_GFDL, DataUi; gc.collect()

    # --- Variables V (Viento Meridional) ---
    V_IPSL = normalize(np.load(os.path.join(BASE_DIR, "VAS_LR_IPSL_Inter1994_2100.npy")))
    V_CNRM = normalize(np.load(os.path.join(BASE_DIR, "VAS_LR_CNRM_Inter1994_2100.npy")))
    V_GFDL = normalize(np.load(os.path.join(BASE_DIR, "VAS_LR_GFDL_Inter1994_2100.npy")))
    
    DataVi = np.concatenate((V_IPSL[:, :, ::SALTO], V_CNRM[:, :, ::SALTO], V_GFDL[:, :, ::SALTO]), axis=2)
    DataV = np.transpose(DataVi, (2, 0, 1))[:, :, :, np.newaxis]
    del V_IPSL, V_CNRM, V_GFDL, DataVi; gc.collect()

    # --- Etiquetas (Alta Resolución) ---
    DS_IPSL = np.load(os.path.join(BASE_DIR, "TOS_DS_IPSL_1994_2100.npy"))[1:298, 1:319, :]
    DS_CNRM = np.load(os.path.join(BASE_DIR, "TOS_DS_CNRM_1994_2100.npy"))[1:298, 1:319, :]
    DS_GFDL = np.load(os.path.join(BASE_DIR, "TOS_DS_GFDL_1994_2100.npy"))[1:298, 1:319, :]
    DS_Ti = np.concatenate((DS_IPSL[:, :, ::SALTO], DS_CNRM[:, :, ::SALTO], DS_GFDL[:, :, ::SALTO]), axis=2)
    
    # --- Máscara ---
    M = np.load(os.path.join(BASE_DIR, "Mascara.npy"))
    DS_Ti = DS_Ti * M[:, :, np.newaxis]
    Label = np.transpose(DS_Ti, (2, 0, 1))[:, :, :, np.newaxis]
    
    del DS_IPSL, DS_CNRM, DS_GFDL, DS_Ti; gc.collect()
    
    logger.info(f"Datos cargados exitosamente. Dimensión temporal: {T}")
    return DataT, DataU, DataV, Label, M, T