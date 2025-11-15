# CSV Exporter Utility

Módulo para exportar `LabeledMotionFeatures` a formato CSV con columnas planas (flatten).

## Características

- ✅ Exporta motion features con todas las columnas en formato plano
- ✅ Soporta exportación a un solo CSV o múltiples CSVs (uno por video)
- ✅ Opción para incluir o excluir frames sin etiquetar
- ✅ Proporciona estadísticas del dataset
- ✅ Compatible con pandas y herramientas de ML

## Estructura del CSV

Cada fila representa un frame procesado con las siguientes columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `video_id` | string | Identificador del video |
| `frame_number` | int | Número de frame en el video |
| `activity_label` | string | Etiqueta de actividad (vacío si no hay etiqueta) |
| `normalized_leg_length` | float | Distancia cadera-tobillo normalizada por longitud de pierna |
| `shoulder_vector_x` | float | Componente X del vector de dirección de hombros (plano XZ) |
| `shoulder_vector_z` | float | Componente Z del vector de dirección de hombros (plano XZ) |
| `ankle_vector_x` | float | Componente X del vector de dirección de tobillos (plano XZ) |
| `ankle_vector_z` | float | Componente Z del vector de dirección de tobillos (plano XZ) |
| `average_hip_angle` | float | Ángulo promedio de caderas (grados) |
| `average_knee_angle` | float | Ángulo promedio de rodillas (grados) |

## Uso Básico

```python
from orchestrator.utils.csv_exporter import CSVExporter
from infrastructure.containers import Container

# Procesar videos con labels
container = Container()
orchestrator = container.motion_orchestrator()

labeled_features = orchestrator.process_videos_with_labels(
    labels_json_path="data/labels/migration.json",
    videos_dir="data/videos",
    target_fps=10.0
)

# Crear exporter
exporter = CSVExporter()

# Exportar a CSV único
exporter.export_to_csv(
    labeled_features_list=labeled_features,
    output_path="output/dataset.csv",
    include_unlabeled=True
)
```

## Métodos Disponibles

### `export_to_csv(labeled_features_list, output_path, include_unlabeled=True)`

Exporta todos los features a un solo archivo CSV.

**Parámetros:**
- `labeled_features_list`: Lista de `LabeledMotionFeatures`
- `output_path`: Ruta del archivo CSV de salida
- `include_unlabeled`: Si incluir frames sin etiqueta (default: `True`)

**Ejemplo:**
```python
exporter.export_to_csv(
    labeled_features_list=features,
    output_path="output/all_data.csv",
    include_unlabeled=False  # Solo exportar frames etiquetados
)
```

### `export_by_video(labeled_features_list, output_dir, include_unlabeled=True)`

Exporta cada video a un archivo CSV separado.

**Parámetros:**
- `labeled_features_list`: Lista de `LabeledMotionFeatures`
- `output_dir`: Directorio para los archivos CSV
- `include_unlabeled`: Si incluir frames sin etiqueta (default: `True`)

**Ejemplo:**
```python
exporter.export_by_video(
    labeled_features_list=features,
    output_dir="output/videos_csv",
    include_unlabeled=True
)
# Genera: output/videos_csv/VIDEO_01.csv, output/videos_csv/VIDEO_02.csv, etc.
```

### `get_statistics(labeled_features_list)`

Obtiene estadísticas del dataset.

**Retorna:**
```python
{
    'total_frames': 1500,
    'labeled_frames': 1200,
    'unlabeled_frames': 300,
    'videos': ['VIDEO_01', 'VIDEO_02'],
    'activity_counts': {
        'walking_towards_camera': 400,
        'standing_still': 300,
        'turning': 200,
        ...
    }
}
```

### `print_statistics(labeled_features_list)`

Imprime estadísticas formateadas del dataset.

```python
exporter.print_statistics(labeled_features)
```

## Uso con Pandas

```python
import pandas as pd

# Cargar CSV
df = pd.read_csv('output/dataset.csv')

# Ver primeras filas
print(df.head())

# Filtrar por actividad
walking = df[df['activity_label'] == 'walking_towards_camera']

# Estadísticas por video
stats = df.groupby('video_id').describe()

# Contar actividades
activity_counts = df['activity_label'].value_counts()
```

## Uso con Machine Learning

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Cargar datos
df = pd.read_csv('output/dataset.csv')

# Preparar features y labels
X = df[['normalized_leg_length', 'shoulder_vector_x', 'shoulder_vector_z',
        'ankle_vector_x', 'ankle_vector_z', 'average_hip_angle',
        'average_knee_angle']]
y = df['activity_label']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluar
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.2%}")
```

## Ejemplos Completos

Ver los archivos de ejemplo:
- `example_process_labeled_videos.py`: Procesamiento completo con exportación
- `example_csv_export.py`: Enfocado en exportación CSV

## Notas

- Los archivos CSV se crean con codificación UTF-8
- Los directorios se crean automáticamente si no existen
- Los frames sin etiqueta tendrán el campo `activity_label` vacío
- Los valores numéricos mantienen su precisión completa en el CSV
