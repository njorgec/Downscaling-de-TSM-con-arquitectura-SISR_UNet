# Downscaling de la temperatura superficial del mar con SISR-UNet

Implementación modular de una red neuronal de super-resolución de imagen única (SISR) con arquitectura U-Net para el downscaling estadístico de la temperatura superficial del mar (TSM) a partir de modelos climáticos globales: **IPSL**, **CNRM** y **GFDL**.

> **Estado del proyecto:** en desarrollo activo. <COMPLETAR: investigación / tesis / trabajo de curso e institución>

![Ejemplo de resultado](docs/images/ejemplo_resultado.png)
<!-- COMPLETAR: figura con tres paneles: baja resolución | predicción | referencia de alta resolución -->

---

## Contenido

1. [Motivación](#motivación)
2. [Datos](#datos)
3. [Arquitectura](#arquitectura)
4. [Estructura del repositorio](#estructura-del-repositorio)
5. [Instalación](#instalación)
6. [Uso](#uso)
7. [Configuración](#configuración)
8. [Resultados](#resultados)
9. [Reproducibilidad](#reproducibilidad)
10. [Limitaciones](#limitaciones)
11. [Cómo citar](#cómo-citar)
12. [Licencia](#licencia)

---

## Motivación

Los modelos climáticos globales entregan la temperatura superficial del mar en una resolución espacial gruesa, insuficiente para estudios regionales como el análisis de eventos extremos o de impactos costeros. El downscaling dinámico exige un costo computacional muy alto; el downscaling estadístico con aprendizaje profundo aprende la relación entre campos de baja y alta resolución a un costo mucho menor.

Este proyecto trata el downscaling como un problema de **super-resolución de imagen única**: una red U-Net recibe un campo de temperatura de baja resolución y reconstruye el campo de alta resolución.

<COMPLETAR: región de estudio, factor de escala (por ejemplo, de X grados a Y grados) y variable objetivo.>

## Datos

| Elemento | Descripción |
| --- | --- |
| Variable | Temperatura superficial del mar |
| Modelos climáticos | IPSL, CNRM, GFDL |
| Referencia de alta resolución | <COMPLETAR: reanálisis u observación utilizada> |
| Periodo de entrenamiento | <COMPLETAR> |
| Periodo de validación | <COMPLETAR> |
| Periodo de prueba | <COMPLETAR> |
| Región | <COMPLETAR: límites de latitud y longitud> |
| Formato | Archivos `.npy` con forma `(tiempo, latitud, longitud)` |

**Obtención de los datos:** <COMPLETAR: enlace o instrucciones de descarga y de preprocesamiento. Si los datos no pueden publicarse, explicar cómo reproducirlos.>

**Normalización:** <COMPLETAR: por ejemplo, estandarización con la media y la desviación estándar calculadas solo con el conjunto de entrenamiento>.

## Arquitectura

<COMPLETAR: diagrama de la red.>

- Codificador y decodificador con conexiones de salto (skip connections).
- <COMPLETAR: número de niveles, filtros por nivel, tipo de interpolación o convolución transpuesta para aumentar la resolución>.
- Función de pérdida: <COMPLETAR>.
- Optimizador y tasa de aprendizaje: <COMPLETAR>.
- Número de parámetros: <COMPLETAR>.

## Estructura del repositorio

```
.
├── SISR_UNet/
│   ├── config.py          # Rutas e hiperparámetros
│   ├── data_utils.py      # Carga y normalización de datos .npy
│   ├── model.py           # Arquitectura U-Net
│   ├── callbacks.py       # Funciones de control durante el entrenamiento
│   ├── utils.py           # Funciones auxiliares
│   ├── train.py           # Script principal de entrenamiento
│   └── requirements.txt   # Dependencias
├── README.md
└── .gitignore
```

## Instalación

Requisitos: Python <COMPLETAR: versión> y, de forma recomendada, una tarjeta gráfica con soporte de CUDA.

```bash
git clone https://github.com/njorgec/Downscaling-de-TSM-con-arquitectura-SISR_UNet.git
cd Downscaling-de-TSM-con-arquitectura-SISR_UNet

python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

pip install -r SISR_UNet/requirements.txt
```

## Uso

### Entrenamiento

```bash
python SISR_UNet/train.py
```

<COMPLETAR: argumentos disponibles, por ejemplo el modelo climático a entrenar, número de épocas y tamaño de lote.>

Antes de ejecutar, edita las rutas de los datos y los hiperparámetros en `SISR_UNet/config.py`.

### Evaluación e inferencia

<COMPLETAR: comando para evaluar un modelo entrenado y generar predicciones.>

## Configuración

| Parámetro | Valor por defecto | Descripción |
| --- | --- | --- |
| <COMPLETAR> | <COMPLETAR> | <COMPLETAR> |

## Resultados

Métricas en el periodo de prueba:

| Modelo climático | Raíz del error cuadrático medio (°C) | Error absoluto medio (°C) | Correlación |
| --- | --- | --- | --- |
| IPSL | <COMPLETAR> | <COMPLETAR> | <COMPLETAR> |
| CNRM | <COMPLETAR> | <COMPLETAR> | <COMPLETAR> |
| GFDL | <COMPLETAR> | <COMPLETAR> | <COMPLETAR> |

Línea base de comparación: <COMPLETAR: por ejemplo, interpolación bicúbica>.

<COMPLETAR: mapas de error espacial y series temporales comparativas.>

## Reproducibilidad

- Semilla aleatoria fija: <COMPLETAR>.
- Versiones exactas de las dependencias en `SISR_UNet/requirements.txt`.
- Pesos entrenados: <COMPLETAR: enlace a las versiones (releases) del repositorio o a otro almacenamiento>.
- Hardware y tiempo de entrenamiento: <COMPLETAR>.

## Limitaciones

- <COMPLETAR: por ejemplo, el modelo se entrenó solo para una región y un factor de escala.>
- <COMPLETAR: sesgos o supuestos conocidos.>

## Cómo citar

Si utilizas este código, cita este repositorio:

```bibtex
@software{<COMPLETAR>,
  author = {<COMPLETAR>},
  title  = {Downscaling de la temperatura superficial del mar con arquitectura SISR-UNet},
  year   = {2026},
  url    = {https://github.com/njorgec/Downscaling-de-TSM-con-arquitectura-SISR_UNet}
}
```

## Licencia

<COMPLETAR: por ejemplo, MIT. Agrega el archivo `LICENSE` en la raíz del repositorio.>

## Autoría y contacto

<COMPLETAR: nombre, institución y medio de contacto.>