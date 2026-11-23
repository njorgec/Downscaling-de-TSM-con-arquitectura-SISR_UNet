# Proyecto: Downscaling Climático con U-Net

Este módulo contiene la implementación de una red neuronal U-Net personalizada diseñada para el *downscaling* (super-resolución) de datos climáticos (temperatura y vientos) provenientes de múltiples modelos (IPSL, CNRM, GFDL).

## Estructura del Proyecto
- `config.py`: Hiperparámetros y rutas.
- `data_utils.py`: Carga y normalización de matrices `.npy`.
- `model.py`: Arquitectura de la red.
- `callbacks.py`: Guardado de emergencia ante interrupciones.
- `utils.py`: Funciones de reproducibilidad.
- `train.py`: Orquestador del entrenamiento.

## Requisitos
1. Clonar el repositorio.
2. Instalar dependencias: `pip install -r requirements.txt`
3. Configurar la ruta de los datos en `config.py` (variable `BASE_DIR`).

## Ejecución
Para iniciar o reanudar automáticamente el entrenamiento:
```bash
python train.py