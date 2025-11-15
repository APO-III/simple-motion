# ML Training Module - Sequence Generation for LSTM

Este módulo convierte features de movimiento frame-por-frame en secuencias temporales para entrenamiento de modelos LSTM.

## 📋 ¿Qué hace este módulo?

Transforma datos de este formato:
```csv
video_id,frame_number,activity_label,normalized_leg_length,shoulder_vector_x,...
girar_lento,1,turning,0.971,-0.844,0.535,...
girar_lento,2,turning,0.971,-0.845,0.533,...
...
```

A este formato:
```python
X_train.shape = (44, 30, 7)  # 44 secuencias, 30 frames cada una, 7 features
y_train.shape = (44,)         # 44 labels (una por secuencia)
```

## 🏗️ Arquitectura (Clean Architecture)

```
ml_training/
├── domain/                   # Modelos de dominio
│   └── sequence.py
│       - MotionSequence      # Una secuencia (30 frames × 7 features)
│       - SequenceDataset     # Colección de secuencias
│       - SequenceGeneratorConfig  # Configuración
│
├── use_cases/                # Lógica de negocio
│   └── sequence_generator.py
│       - SequenceGenerator   # Genera ventanas deslizantes desde CSV
│
└── utils/                    # Utilidades
    ├── data_splitter.py
    │   - DataSplitter        # Split train/val/test a nivel de VIDEO
    └── label_encoder.py
        - LabelEncoder        # Convierte labels ↔ índices
```

## 🚀 Uso Rápido

### 1. Generar Secuencias

```python
from ml_training.domain.sequence import SequenceGeneratorConfig
from ml_training.use_cases.sequence_generator import SequenceGenerator

# Configurar
config = SequenceGeneratorConfig(
    window_size=30,    # 30 frames por secuencia (1 segundo @ 30 FPS)
    stride=15,         # 50% overlap entre ventanas
)

# Generar desde CSV
generator = SequenceGenerator(config=config)
dataset = generator.generate_from_csv(
    csv_path="results/raw/source3.csv",
    source_name="source3"
)

# O desde múltiples sources
dataset = generator.generate_from_multiple_csvs([
    ("results/raw/source1.csv", "source1"),
    ("results/raw/source3.csv", "source3"),
])

# Ver estadísticas
dataset.print_statistics()
# Output:
# Total sequences: 91
# Sequence shape: (30, 7)
# Number of videos: 11
# Number of classes: 7
```

### 2. Split Train/Val/Test

```python
from ml_training.utils.data_splitter import DataSplitter

splitter = DataSplitter(random_seed=42)

train_dataset, val_dataset, test_dataset = splitter.split_by_video(
    dataset=dataset,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify_by_label=True
)
```

**IMPORTANTE:** El split se hace a nivel de **VIDEO**, no de secuencias individuales. Esto previene data leakage (secuencias del mismo video en train y test).

### 3. Obtener Arrays para Entrenamiento

```python
# Arrays NumPy listos para LSTM
X_train = train_dataset.get_X()  # shape: (num_sequences, 30, 7)
y_train = train_dataset.get_y()  # shape: (num_sequences,)

# Para categorical crossentropy
num_classes = len(dataset.label_to_index)
y_train_cat = train_dataset.get_y_categorical(num_classes)  # shape: (num_sequences, 7)
```

## 📊 Conceptos Clave

### Ventanas Deslizantes (Sliding Windows)

```
Video con 109 frames, actividad: "standing_up"

Con window_size=30, stride=15:

Ventana 1:  frames [1-30]    → Secuencia 1
            └─────────┘
Ventana 2:  frames [16-45]   → Secuencia 2
                └─────────┘
Ventana 3:  frames [31-60]   → Secuencia 3
                    └─────────┘
...

Resultado: 7 secuencias de shape (30, 7) desde 109 frames
```

### Manejo de Cambios de Label

Si un video tiene múltiples actividades:

```
Video "girar_lento":
  Frames 1-140:   "turning"        → Genera ventanas aquí
  Frames 141-151: "standing_still" → Solo 11 frames, descartado (< 30)
```

El generador:
1. **Detecta automáticamente** cambios de label
2. **Crea segmentos** por actividad
3. **Descarta segmentos** más cortos que `window_size`
4. **Genera ventanas SOLO dentro** de cada segmento (nunca mezcla labels)

## 📈 Resultado con tus Datos

Ejecutando `example_generate_sequences.py` con source1 + source3:

```
✓ Total sequences: 91
✓ Sequence shape: (30, 7)
✓ Number of videos: 11
✓ Number of classes: 7

Class distribution:
  sitting_down            : 9 sequences (9.89%)
  sitting_still           : 10 sequences (10.99%)
  standing_still          : 3 sequences (3.30%)
  standing_up             : 8 sequences (8.79%)
  turning                 : 19 sequences (20.88%)
  walking_away_from_camera: 19 sequences (20.88%)
  walking_towards_camera  : 23 sequences (25.27%)

Split:
  Train: 44 sequences from 6 videos
  Test:  47 sequences from 5 videos
```

## 🔧 Configuración

### SequenceGeneratorConfig

```python
config = SequenceGeneratorConfig(
    window_size=30,           # Frames por secuencia
    stride=15,                # Salto entre ventanas
    min_segment_length=30,    # Mínimo de frames para procesar un segmento
    feature_columns=[         # Columnas del CSV a usar
        "normalized_leg_length",
        "shoulder_vector_x",
        "shoulder_vector_z",
        "ankle_vector_x",
        "ankle_vector_z",
        "average_hip_angle",
        "average_knee_angle"
    ]
)
```

**Recomendaciones:**
- `window_size=30` → 1 segundo @ 30 FPS (captura movimientos completos)
- `stride=15` → 50% overlap (más datos, transiciones suaves)
- `stride=10` → 66% overlap (aún más datos si tienes pocos videos)

## 📁 Archivos de Salida

### label_encoder.json

```json
{
  "label_to_index": {
    "sitting_down": 0,
    "standing_up": 3,
    "turning": 4,
    ...
  },
  "index_to_label": {
    "0": "sitting_down",
    "3": "standing_up",
    ...
  },
  "num_classes": 7
}
```

**Uso:** Cargar este archivo durante inferencia para decodificar predicciones del modelo.

```python
from ml_training.utils.label_encoder import LabelEncoder

encoder = LabelEncoder.load("output/label_encoder.json")
predicted_index = 4
activity = encoder.decode(predicted_index)  # "turning"
```

## ⚠️ Consideraciones Importantes

### 1. Dataset Pequeño

Con solo 11 videos y 91 secuencias:
- El split 70/15/15 puede resultar en 0 videos para validación
- **Solución:** Usar más videos de source2, o ajustar ratios (80/10/10)

### 2. Desbalance de Clases

```
standing_still: 3 sequences (3.30%)   ← MUY POCO
turning:        19 sequences (20.88%)  ← OK
```

**Soluciones:**
- Procesar más videos con actividades poco frecuentes
- Usar `class_weights` en el entrenamiento LSTM
- Data augmentation (variaciones de velocidad, ruido)

### 3. Frames Descartados

El generador descarta:
- Segmentos < 30 frames (muy cortos)
- Frames en transiciones entre actividades

```
⚠ Skipped segment [1-13] 'sitting_down': too short (13 frames)
⚠ Skipped segment [264-290] 'turning': too short (27 frames)
```

**Esto es correcto** para mantener pureza de labels, pero significa que pierdes ~20-30% de frames.

## 🎯 Próximos Pasos

1. **Procesar source2** para tener más datos
2. **Implementar LSTM trainer** en `ml_training/use_cases/lstm_trainer.py`
3. **Definir arquitectura LSTM** en `ml_training/infrastructure/keras_lstm_model.py`
4. **Entrenar modelo** con las secuencias generadas

## 📖 Ejemplos

Ver `example_generate_sequences.py` para un ejemplo completo de uso.

```bash
python example_generate_sequences.py
```

## 🧪 Tests

```bash
pytest ml_training/tests/
```

(Nota: Tests pendientes de implementar)
