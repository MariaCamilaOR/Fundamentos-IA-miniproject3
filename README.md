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
- `best_pokemon_resnet50_finetuned.keras`: Modelo fine-tuned
- `README.md`: Este archivo con explicaciones detalladas

## Cómo ejecutar

1. Instalar las dependencias necesarias (se instalan automáticamente en el notebook)
2. Ejecutar las celdas del notebook en orden
3. El modelo se entrenará y evaluará automáticamente
4. Ejecutar la celda de Fine-Tuning para mejorar el modelo

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

## Celda 13: Definición del modelo con Transfer Learning

Sigue el esquema de Transfer Learning:

1. **Carga ResNet50** preentrenada en ImageNet sin la parte densa superior:
   - `include_top=False`
   - `weights='imagenet'`

2. **Congela** sus capas (Feature Extraction) al inicio para no modificar los pesos originales

3. **Añade una "cabeza" de clasificación:**
   - `GlobalAveragePooling2D`: reduce el mapa de características a un vector
   - `Dropout(0.3)`: regularización para evitar overfitting
   - `Dense(num_classes, activation='softmax')`: clasifica en 15 Pokémon

**Estrategia:**
- Primero se entrena solo la cabeza (capas nuevas) con las capas convolucionales congeladas
- Luego, opcionalmente, se puede hacer fine-tuning descongelando algunas capas superiores

---

## Celda 15: Construcción del modelo ResNet50

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

## Celda 16: Compilación y entrenamiento (Feature Extraction)

**Compilación:**
- Optimizador: Adam con learning rate 1e-3
- Loss: `sparse_categorical_crossentropy` (multiclase)
- Métricas: Accuracy, Top-3 Accuracy, Top-5 Accuracy

**Callbacks:**
- **EarlyStopping**: 
  - Monitorea `val_top_3_accuracy`
  - Patience de 5 épocas
  - Restaura los mejores pesos al finalizar
- **ModelCheckpoint**: 
  - Guarda el mejor modelo en `best_pokemon_resnet50_feature_extraction.keras`
  - Solo guarda cuando mejora `val_top_3_accuracy`

**Entrenamiento:**
- Máximo 30 épocas (EarlyStopping puede parar antes)
- Solo se entrenan las capas nuevas (cabeza de clasificación)
- Las capas de ResNet50 permanecen congeladas

---

## Celda 18: Carga del mejor modelo

Carga el mejor modelo guardado por ModelCheckpoint desde el checkpoint.

**Importante:** A partir de aquí, se usa `best_model` en lugar de `model` para todas las evaluaciones y predicciones, ya que representa el mejor estado del modelo durante el entrenamiento.

---

## Celda 19: Fine-Tuning

Fine-tuning de las capas superiores de ResNet50 usando el mejor modelo guardado.

**Proceso:**

1. **Obtiene la base convolucional** dentro de `best_model`
2. **Descongela solo las últimas 30 capas** de ResNet50 (aproximadamente el último bloque)
3. **Recompila** con un learning rate más bajo (1e-5) para ajuste fino
4. **Entrena** con callbacks similares (EarlyStopping y ModelCheckpoint)
5. **Evalúa** el modelo afinado
6. **Visualiza** las curvas de entrenamiento del fine-tuning

**Ventajas del fine-tuning:**
- Adapta las características aprendidas al dominio específico de Pokémon
- Puede mejorar algunos puntos porcentuales la accuracy
- Se hace de forma conservadora (solo últimas capas, LR bajo)

---

## Celda 22: Evaluación del modelo

Evalúa el modelo usando métricas apropiadas para CNN con muchas clases:

### Métricas principales:

1. **Top-K Accuracy** (Top-3, Top-5): 
   - Estándar en visión por computadora
   - Mide si la clase correcta está entre las K predicciones más probables

2. **Macro-averaged F1-score**: 
   - Promedio no ponderado de F1 por clase
   - Útil cuando hay desbalance entre clases

3. **Micro-averaged F1-score**: 
   - F1 calculado globalmente
   - Equivalente a accuracy cuando hay balance

4. **Balanced Accuracy**: 
   - Promedio del recall por clase
   - Mejor que accuracy cuando hay clases desbalanceadas

### Proceso de evaluación:

1. Evalúa el modelo en `val_ds` para obtener métricas
2. Extrae predicciones y etiquetas verdaderas
3. Calcula todas las métricas mencionadas
4. Genera matriz de confusión (numérica y gráfica con heatmap)
5. Genera reporte de clasificación por clase (precision, recall, F1)

---

## Celda 22: Visualización de predicciones

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

## Celda 26: Conclusiones

Resumen del mini proyecto:

- Se aplicaron conceptos de Redes Neuronales Artificiales profundas y CNN
- Se utilizó Transfer Learning con ResNet50 preentrenada en ImageNet
- Se aplicó Data Augmentation para aumentar la robustez
- Se entrenó en esquema de Feature Extraction y Fine-Tuning
- Se evaluó con métricas apropiadas para CNN multiclase

**Objetivos cumplidos:**
- Implementar clasificación supervisada multiclase usando RNA
- Justificar el uso de CNN y Transfer Learning
- Demostrar el proceso completo: carga de datos, preprocesamiento, diseño del modelo, entrenamiento, evaluación y análisis

---

