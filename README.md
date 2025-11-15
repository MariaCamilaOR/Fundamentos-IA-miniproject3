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

## Métricas utilizadas

El proyecto utiliza métricas apropiadas para CNN con clasificación multiclase:

- **Top-K Accuracy** (Top-3, Top-5): Estándar en visión por computadora
- **Macro-averaged F1-score**: Promedio no ponderado de F1 por clase
- **Micro-averaged F1-score**: F1 calculado globalmente
- **Balanced Accuracy**: Promedio del recall por clase

## Estructura del proyecto

- `miniproyecto3_pokemon.ipynb`: Notebook principal con todo el código
- `best_pokemon_resnet50_feature_extraction.keras`: Modelo entrenado guardado
- `README.md`: Este archivo

## Cómo ejecutar

1. Instalar las dependencias necesarias (se instalan automáticamente en el notebook)
2. Ejecutar las celdas del notebook en orden
3. El modelo se entrenará y evaluará automáticamente

## Dataset

Dataset utilizado: [7,000 Labeled Pokémon](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) de Kaggle.

