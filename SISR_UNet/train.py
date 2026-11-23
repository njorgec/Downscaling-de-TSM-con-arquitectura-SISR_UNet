# train.py
"""
Script principal para el entrenamiento del modelo U-Net.
"""
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from config import CHECKPOINT_DIR, FINAL_MODEL_PATH, EPOCHS, SEED, VALIDATION_SPLIT, LEARNING_RATE
from data_utils import clear_memory, load_dataset
from model import build_unet
from callbacks import EpochSaverCallback, EmergencySaveCallback
from utils import setup_reproducibility, find_latest_checkpoint

# Configuración del sistema de logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== INICIANDO PIPELINE DE ENTRENAMIENTO ===")
    
    # 1. Preparación del entorno
    clear_memory()
    setup_reproducibility(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 2. Carga de datos
    DataT, DataU, DataV, Label, M, T = load_dataset()
    
    # Expandir dimensión de la máscara para que coincida con el batch
    M_expanded = np.repeat(M[np.newaxis, :, :, np.newaxis], T, axis=0)

    # 3. Construcción del modelo
    model = build_unet(seed=SEED)

    # 4. Configuración de Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "cp-{epoch:04d}.keras"),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        EmergencySaveCallback(CHECKPOINT_DIR),
        EpochSaverCallback(CHECKPOINT_DIR)
    ]

    # 5. Reanudar o iniciar nuevo entrenamiento
    latest_checkpoint = find_latest_checkpoint()
    initial_epoch = 0

    if latest_checkpoint:
        logger.info(f"Cargando checkpoint: {latest_checkpoint}")
        model = tf.keras.models.load_model(latest_checkpoint)
        
        if "emergency" in latest_checkpoint:
            try:
                with open(os.path.join(CHECKPOINT_DIR, "last_epoch.txt"), "r") as f:
                    initial_epoch = int(f.read().strip()) + 1
            except Exception as e:
                logger.warning(f"No se pudo leer la época de emergencia: {e}")
                initial_epoch = 0
        else:
            initial_epoch = int(latest_checkpoint.split('-')[-1].split('.')[0]) + 1
            
        logger.info(f"Reanudando entrenamiento desde la época {initial_epoch}")
    else:
        logger.info("No se encontraron checkpoints. Iniciando entrenamiento nuevo.")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=['mae', 'mse']
        )

    # 6. Entrenamiento
    logger.info("Iniciando model.fit()...")
    history = model.fit(
        {"T": DataT, "U": DataU, "V": DataV, "Mask": M_expanded}, 
        Label,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )

    # 7. Guardado final
    logger.info(f"Guardando modelo final en: {FINAL_MODEL_PATH}")
    model.save(FINAL_MODEL_PATH)
    logger.info("=== ENTRENAMIENTO FINALIZADO CON ÉXITO ===")

if __name__ == "__main__":
    main()