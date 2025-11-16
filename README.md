# Fundamentos-IA-miniproject3

Mini Proyecto 3 – Clasificación de Pokémon con Transfer Learning (ResNet50)

Este proyecto implementa un clasificador de imágenes para **15 Pokémon** seleccionados de la generación 1 usando el dataset **"7,000 Labeled Pokémon"** de Kaggle.

## Objetivo

Entrenar y evaluar un modelo de **clasificación supervisada multiclase** que, a partir de una imagen, prediga correctamente a qué Pokémon pertenece (entre 15 clases seleccionadas).

## Tecnologías utilizadas

- **TensorFlow/Keras**: Para la construcción y entrenamiento del modelo
- **ResNet50**: Modelo preentrenado en ImageNet para Transfer Learning
- **Kaggle Hub**: Para descargar el dataset
- **scikit-learn**: Para métricas de evaluación
- **seaborn**: Para visualización de matrices de confusión

## Métricas utilizadas

El proyecto utiliza métricas apropiadas para CNN con clasificación multiclase:

- **Top-K Accuracy** (Top-3, Top-5): Estándar en visión por computadora
- **Macro-averaged F1-score**: Promedio no ponderado de F1 por clase
- **Micro-averaged F1-score**: F1 calculado globalmente
- **Balanced Accuracy**: Promedio del recall por clase
- **Matriz de confusión**: Para analizar confusiones entre clases
- **Reporte de clasificación**: Precision, recall, F1 por clase

## Estructura del proyecto

- `miniproyecto3_pokemon.ipynb`: Notebook principal con todo el código
- `best_pokemon_resnet50_feature_extraction.keras`: Modelo entrenado guardado (Feature Extraction)
- `README.md`: Este archivo con explicaciones detalladas

## Cómo ejecutar

1. Instalar las dependencias necesarias (se instalan automáticamente en el notebook)
2. Ejecutar las celdas del notebook en orden
3. El modelo se entrenará y evaluará automáticamente
4. El mejor modelo se guardará automáticamente usando ModelCheckpoint

## Dataset

Dataset utilizado: [7,000 Labeled Pokémon](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) de Kaggle.

---

## Explicación detallada por celdas

## Celda 1: Introducción al Proyecto

Este cuaderno implementa un clasificador de imágenes para **15 Pokémon** seleccionados de la generación 1 usando el dataset **"7,000 Labeled Pokémon"** de Kaggle (`lantian773030/pokemonclassification`).

### Objetivo del mini proyecto

Entrenar y evaluar un modelo de **clasificación supervisada multiclase** que, a partir de una imagen, prediga correctamente a qué Pokémon pertenece (entre 15 clases seleccionadas).

### Temas aplicados

- **Neuronas artificiales y capas densas** (funciones de activación, suma ponderada y bias)
- **Redes neuronales profundas (Deep Learning)** (problemas de muchas capas, gradiente, necesidad de mucha data)
- **Redes Convolucionales (CNN)** (capas convolucionales con kernels/filtros, ReLU, pooling, flatten y capas fully-connected)
- **Entrenamiento de RNA con Backpropagation** (ajuste de pesos mediante gradiente descendente)
- **Transfer Learning** (reutilización de un modelo preentrenado en ImageNet como extractor de características)

### Enfoque general

1. Descargar el dataset de Pokémon con `kagglehub`
2. Cargar las imágenes desde carpetas (una carpeta por clase)
3. Aplicar preprocesamiento (redimensionar a 224×224, normalización, data augmentation)
4. Definir un modelo de Transfer Learning con ResNet50 (Feature Extraction)
5. Entrenar y evaluar con métricas apropiadas para CNN multiclase

---

## Celda 2: Referencias

Referencia al repositorio original: https://github.com/deeplearningunb/pokedex

---

## Celda 3: Instalación de dependencias

Instala todas las dependencias necesarias:
- TensorFlow: Framework de deep learning
- kagglehub: Para descargar datasets de Kaggle
- scikit-learn: Para métricas de evaluación
- matplotlib: Para visualizaciones
- numpy: Para operaciones numéricas

---

## Celda 4: Imports

Importa todas las librerías necesarias:
- TensorFlow/Keras para el modelo
- sklearn.metrics para evaluación (confusion_matrix, classification_report, f1_score, balanced_accuracy_score, top_k_accuracy_score)
- seaborn para visualización de la matriz de confusión
- numpy, matplotlib, pathlib para manipulación de datos y visualización

Se fija una semilla (SEED=42) para reproducibilidad.

---

## Celda 5: Descarga del dataset

Descarga el dataset completo de Pokémon desde Kaggle usando `kagglehub`.

**Proceso:**
1. Descarga el dataset completo (150 clases disponibles)
2. Selecciona solo las primeras 15 clases (requisito del proyecto)
3. Crea un directorio temporal con solo las 15 clases seleccionadas
4. Copia las carpetas de las clases seleccionadas

Las 15 clases seleccionadas son: Abra, Aerodactyl, Alakazam, Alolan Sandslash, Arbok, Arcanine, Articuno, Beedrill, Bellsprout, Blastoise, Bulbasaur, Butterfree, Caterpie, Chansey, Charizard.

---

## Celda 7: Creación de datasets de entrenamiento y validación

Usa `tf.keras.utils.image_dataset_from_directory` para:

- Leer las imágenes directamente desde las carpetas
- Crear automáticamente etiquetas a partir de los nombres de carpeta
- Redimensionar a `224×224` (requisito de entrada de ResNet50)
- Dividir el dataset en:
  - **80%** para entrenamiento (`subset="training"`)
  - **20%** para validación (`subset="validation"`)

**Parámetros importantes:**
- `seed=SEED`: asegura que la partición train/valid sea reproducible
- `batch_size=32`: tamaño de lote estándar
- `image_size=(224, 224)`: tamaño de entrada compatible con ResNet50

**Optimizaciones:**
- `cache()`: guarda las imágenes en memoria después de la primera lectura
- `shuffle(1000)`: mezcla los datos de entrenamiento
- `prefetch()`: precarga batches mientras el modelo entrena

---

## Celda 9: Visualización de ejemplos de entrenamiento

Muestra una cuadrícula de 3×3 con imágenes del dataset de entrenamiento junto con sus etiquetas.

**Propósito:**
- Verificar que las imágenes se cargan correctamente
- Verificar que las etiquetas (`class_names`) sean correctas
- Tener una idea de la variabilidad de las imágenes (fondo, escala, posición del Pokémon)

---

## Celda 11: Data Augmentation

Define una capa de data augmentation con transformaciones aleatorias:

- **RandomFlip("horizontal")**: Reflejo horizontal aleatorio
- **RandomRotation(0.1)**: Rotación aleatoria hasta 10%
- **RandomZoom(0.1)**: Zoom aleatorio hasta 10%

**Propósito:**
- Aumentar la variabilidad de las imágenes
- Reducir el riesgo de overfitting cuando hay poca data por clase
- Mejorar la generalización del modelo

Esta capa se aplica dentro del modelo durante el entrenamiento.

---

## Celda 13: Definición y construcción del modelo ResNet50

Esta celda explica el enfoque de Transfer Learning y la construcción del modelo.

**Estrategia de Transfer Learning:**

1. **Carga ResNet50** preentrenada en ImageNet sin la parte densa superior:
   - `include_top=False`
   - `weights='imagenet'`

2. **Congela** sus capas (Feature Extraction) al inicio para no modificar los pesos originales

3. **Añade una "cabeza" de clasificación:**
   - `GlobalAveragePooling2D`: reduce el mapa de características a un vector
   - `Dropout(0.3)`: regularización para evitar overfitting
   - `Dense(num_classes, activation='softmax')`: clasifica en 15 Pokémon

**Estrategia:**
- Se entrena solo la cabeza (capas nuevas) con las capas convolucionales congeladas
- Feature Extraction: las características aprendidas en ImageNet se reutilizan para Pokémon

---

## Celda 14: Construcción del modelo ResNet50

Construye el modelo completo con:

1. **Input**: Imágenes de tamaño 224×224×3
2. **Data augmentation**: Transformaciones aleatorias
3. **Preprocesamiento**: `preprocess_input` específico de ResNet50
4. **Base convolucional**: ResNet50 congelada (Feature Extraction)
5. **Pooling global**: Reduce dimensionalidad
6. **Dropout**: Regularización
7. **Capa densa final**: Clasificación en 15 clases con softmax

El modelo se llama `pokemon_resnet50` y muestra un resumen con `model.summary()`.

---

## Celda 15: Compilación y entrenamiento

**Compilación:**
- Optimizador: Adam con learning rate 1e-3
- Loss: `sparse_categorical_crossentropy` (multiclase)
- Métricas: Accuracy, Top-3 Accuracy, Top-5 Accuracy

**Callbacks:**
- **EarlyStopping**: 
  - Monitorea `val_accuracy`
  - Patience de 3 épocas
  - Restaura los mejores pesos al finalizar
- **ModelCheckpoint**: 
  - Guarda el mejor modelo en `best_pokemon_resnet50_feature_extraction.keras`
  - Solo guarda cuando mejora `val_accuracy`

**Entrenamiento:**
- Máximo 15 épocas (EarlyStopping puede parar antes si no hay mejora)
- Solo se entrenan las capas nuevas (cabeza de clasificación)
- Las capas de ResNet50 permanecen congeladas (Feature Extraction)

---

## Celda 17: Curvas de entrenamiento

Gráficas de Accuracy y Loss durante el entrenamiento para:
- Detectar overfitting (diferencia entre train y validation)
- Verificar si el error se desborda
- Analizar la evolución del modelo en entrenamiento vs validación

**Análisis automático:**
- Compara accuracy final de train vs validation
- Compara loss final de train vs validation
- Muestra advertencias si hay overfitting o desbordamiento de error

---

## Celda 18: Carga del mejor modelo

Carga el mejor modelo guardado por ModelCheckpoint desde el checkpoint.

**Importante:** A partir de aquí, se usa `best_model` en lugar de `model` para todas las evaluaciones y predicciones, ya que representa el mejor estado del modelo durante el entrenamiento.

---

## Celda 20: Evaluación del modelo - Top-K Accuracy

Evalúa el modelo usando métricas de Top-K Accuracy (Top-1, Top-3, Top-5) porque son estándar en visión por computadora y permiten evaluar si la clase correcta está entre las K predicciones más probables.

---

## Celda 21: Evaluación global y obtención de predicciones

Obtiene las predicciones del modelo y las etiquetas verdaderas para cálculos posteriores.

---

## Celda 22: Métricas adicionales - F1-score y Balanced Accuracy

Calcula métricas adicionales apropiadas para CNN multiclase:
- Macro y Micro F1-score para evaluar el desempeño considerando el desbalance de clases
- Balanced Accuracy que es más robusta que accuracy cuando hay clases desbalanceadas

---

## Celda 24: Matriz de confusión

La matriz de confusión muestra los errores de clasificación por clase, permitiendo identificar qué Pokémon se confunden entre sí. Se muestra tanto en valores absolutos como normalizada (porcentajes).

---

## Celda 26: Classification Report

El reporte de clasificación muestra precision, recall y F1-score por clase, permitiendo un análisis detallado del desempeño del modelo para cada Pokémon.

---

## Celda 28: Curvas ROC y AUC

Las curvas ROC (Receiver Operating Characteristic) y el área bajo la curva (AUC) permiten evaluar el desempeño del modelo en clasificación multiclase usando el enfoque One-vs-Rest. El AUC mide la capacidad del modelo para distinguir entre clases, siendo 1.0 el valor perfecto.

---

## Celda 30: Visualización de predicciones

Esta celda explica el propósito de la visualización de predicciones.

## Celda 31: Visualización de predicciones vs etiquetas reales

Muestra una cuadrícula de 3×3 con imágenes del conjunto de validación junto con:
- **Predicción del modelo**: Nombre del Pokémon predicho
- **Etiqueta real**: Nombre del Pokémon verdadero

**Visualización:**
- Títulos en verde si la predicción es correcta
- Títulos en rojo si la predicción es incorrecta

Permite identificar visualmente:
- Las confusiones más comunes entre clases
- Qué Pokémon se clasifican mejor o peor
- La calidad general de las predicciones

---

## Conclusiones

Resumen del mini proyecto:

- Se aplicaron conceptos de Redes Neuronales Artificiales profundas y CNN
- Se utilizó Transfer Learning con ResNet50 preentrenada en ImageNet
- Se aplicó Data Augmentation para aumentar la robustez
- Se entrenó en esquema de Feature Extraction
- Se evaluó con métricas apropiadas para CNN multiclase:
  - Top-K Accuracy (Top-1, Top-3, Top-5)
  - Macro y Micro F1-score
  - Balanced Accuracy
  - Matriz de confusión (normalizada)
  - Classification Report
  - Curvas ROC y AUC

**Objetivos cumplidos:**
- Implementar clasificación supervisada multiclase usando RNA
- Justificar el uso de CNN y Transfer Learning
- Demostrar el proceso completo: carga de datos, preprocesamiento, diseño del modelo, entrenamiento, evaluación y análisis
- Visualizar curvas de entrenamiento para detectar overfitting
- Generar visualizaciones completas de evaluación (matriz de confusión, ROC-AUC)

---

