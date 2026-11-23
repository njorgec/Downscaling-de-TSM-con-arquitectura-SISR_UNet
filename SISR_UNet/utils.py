# utils.py
"""
Funciones auxiliares para reproducibilidad y manejo de checkpoints.
"""
import os
import glob
import random
import logging
import numpy as np
import tensorflow as tf
from config import CHECKPOINT_DIR

logger = logging.getLogger(__name__)

def setup_reproducibility(seed: int = 42) -> None:
    """Configura semillas para garantizar la reproducibilidad del entrenamiento."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    logger.info(f"Reproducibilidad configurada con semilla {seed}.")

def find_latest_checkpoint() -> str | None:
    """
    Busca el checkpoint más reciente o un guardado de emergencia.
    
    Returns:
        str | None: Ruta al archivo .keras o None si no existe.
    """
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "cp-*.keras"))
    if files:
        epochs = [int(f.split('-')[-1].split('.')[0]) for f in files]
        latest = max(epochs)
        return os.path.join(CHECKPOINT_DIR, f"cp-{latest:04d}.keras")
    
    emergency_file = os.path.join(CHECKPOINT_DIR, "emergency_save.keras")
    if os.path.exists(emergency_file):
        return emergency_file
        
    return None