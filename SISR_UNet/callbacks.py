# callbacks.py
"""
Callbacks personalizados para el guardado de emergencia y registro de épocas.
"""
import os
import signal
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

class EpochSaverCallback(tf.keras.callbacks.Callback):
    """Guarda el número de la época actual en un archivo de texto."""
    def __init__(self, checkpoint_dir: str):
        super().__init__()
        self.epoch_file = os.path.join(checkpoint_dir, "last_epoch.txt")

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        with open(self.epoch_file, "w") as f:
            f.write(str(epoch))

class EmergencySaveCallback(tf.keras.callbacks.Callback):
    """Atrapa señales del sistema (Ctrl+C, kill) para guardar el modelo antes de cerrar."""
    def __init__(self, checkpoint_dir: str):
        super().__init__()
        self.emergency_file = os.path.join(checkpoint_dir, "emergency_save.keras")

    def on_train_begin(self, logs: dict = None) -> None:
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, signum: int, frame) -> None:
        logger.warning(f"Señal {signum} recibida. Guardando modelo de emergencia...")
        self.model.save(self.emergency_file)
        logger.info(f"Modelo de emergencia guardado en: {self.emergency_file}")
        exit(0)