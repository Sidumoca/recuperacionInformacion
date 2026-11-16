#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extractor y clasificador de TFG de Zaguán (Universidad de Zaragoza).

Uso:
    python clasificadorTexto.py -dir <carpeta_xml> -output <carpeta_resultados>

El programa:
  1. Recorre recursivamente los XML.
  2. Conserva únicamente los registros dc:type = TAZ-TFG.
  3. Extrae título, descripción y titulación (dc:subject).
  4. Asigna la titulación a una de las seis categorías.
  5. Guarda datos_clasificados.csv y resumen_extraccion.txt.
  6. Entrena una LSTM y guarda precision.txt, confusion.txt y error.jpg.

Dependencias:
    pip install tensorflow scikit-learn pandas numpy matplotlib
"""

import argparse
import csv
import os
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.layers import TextVectorization


# ---------------------------------------------------------------------
# Configuración reproducible
# ---------------------------------------------------------------------

SEED = 0
EPOCHS = 30
BATCH_SIZE = 64
MAX_TOKENS = 20000
MAX_SEQUENCE_LENGTH = 200

CATEGORIES = [
    "Ciencias de la Salud",
    "Ciencias",
    "Ingeniería y Arquitectura",
    "Filosofia y Letras",
    "Economía y Empresa",
    "Derecho, Educación y Ciencias Sociales",
]


def set_random_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ---------------------------------------------------------------------
# Normalización y extracción XML
# ---------------------------------------------------------------------

def normalize_text(text):
    """Minúsculas, sin tildes y con espacios normalizados."""
    text = (text or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text


def local_name(tag):
    """Obtiene el nombre local de una etiqueta XML con o sin namespace."""
    return tag.rsplit("}", 1)[-1].split(":")[-1]


def get_values(root, field):
    values = []
    for element in root.iter():
        if local_name(element.tag) == field and element.text:
            value = element.text.strip()
            if value:
                values.append(value)
    return values


def choose_language_value(elements, field):
    """Prefiere el valor sin idioma o en español cuando hay títulos repetidos."""
    candidates = []
    for element in elements:
        if local_name(element.tag) == field and element.text and element.text.strip():
            lang = ""
            for key, value in element.attrib.items():
                if local_name(key) == "lang":
                    lang = normalize_text(value)
            candidates.append((lang, element.text.strip()))

    for lang, value in candidates:
        if lang in ("", "es", "spa", "es-es"):
            return value
    return candidates[0][1] if candidates else ""


# Cada regla se aplica sobre el texto de los dc:subject.
# Es deliberadamente explícita: si no reconocemos una titulación, no inventamos
# una categoría. Las reglas se pueden ampliar al revisar el CSV de no clasificados.
DEGREE_RULES = [
    ("Ciencias de la Salud", [
        "enfermer", "fisioter", "medicina", "odontolog", "veterin",
        "terapia ocupacional", "logopedia", "podologia", "farmacia",
        "nutricion humana", "ciencias de la actividad fisica y del deporte",
        "ciencias de la salud", "optica y optometria",
    ]),
    ("Ciencias", [
        "matematic", "fisica", "quimica", "biolog", "geolog",
        "ciencias ambientales", "bioquimica", "biotecnolog",
        "estadistica", "ciencia y tecnologia de los alimentos",
    ]),
    ("Ingeniería y Arquitectura", [
        "ingenier", "arquitect", "edificacion", "obra publica",
        "diseno industrial", "ingeniero tecnico", "ingeniero industrial",
    ]),
    ("Economía y Empresa", [
        "administracion y direccion de empresas",
        "economia", "finanzas", "contabilidad", "marketing",
        "gestion y administracion publica", "relaciones laborales y recursos humanos",
    ]),
    ("Filosofia y Letras", [
        "filologia", "filosofia", "historia", "historia del arte",
        "geografia", "humanidades", "traduccion", "estudios ingleses",
        "estudios clasicos", "lenguas modernas", "bellas artes",
        "informacion y documentacion",
    ]),
    ("Derecho, Educación y Ciencias Sociales", [
        "derecho", "maestro", "educacion", "magisterio", "pedagog",
        "trabajo social", "sociologia", "ciencias politicas",
        "criminologia", "periodismo", "comunicacion audiovisual",
        "psicologia", "relaciones laborales", "turismo",
    ]),
]


def classify_degree(subjects):
    """
    Devuelve una categoría a partir de dc:subject.
    Da prioridad a subjects que parecen nombrar una titulación.
    Si hay categorías incompatibles y no se puede resolver, devuelve None.
    """
    normalized = [(s, normalize_text(s)) for s in subjects]

    degree_markers = (
        "graduado", "graduada", "grado en", "ingeniero", "ingeniera",
        "licenciado", "licenciada", "diplomado", "diplomada",
        "maestro", "maestra", "arquitecto", "arquitecta",
    )

    marked = [(raw, norm) for raw, norm in normalized
              if any(marker in norm for marker in degree_markers)]
    candidates = marked if marked else normalized

    found = []
    for raw, norm in candidates:
        for category, terms in DEGREE_RULES:
            if any(term in norm for term in terms):
                found.append((category, raw, norm))
                break

    distinct = {item[0] for item in found}
    if len(distinct) == 1:
        return found[0][0]

    # Si hay más de una coincidencia, intentamos quedarnos con la titulación
    # más explícita (normalmente el subject más largo que contiene el marcador).
    if len(distinct) > 1:
        marked_found = [
            item for item in found
            if any(marker in item[2] for marker in degree_markers)
        ]
        marked_categories = {item[0] for item in marked_found}
        if len(marked_categories) == 1:
            return marked_found[0][0]

    return None


def extract_record(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    types = get_values(root, "type")
    if "TAZ-TFG" not in types:
        return None, "no_es_tfg"

    title = choose_language_value(root.iter(), "title")
    description = choose_language_value(root.iter(), "description")
    subjects = get_values(root, "subject")

    if not title:
        return None, "sin_titulo"
    if not description:
        return None, "sin_descripcion"

    category = classify_degree(subjects)
    if category is None:
        return {
            "Title": title,
            "Description": description,
            "Class": "",
            "Subjects": " | ".join(subjects),
            "XML": os.path.basename(xml_path),
        }, "tfg_sin_categoria"

    return {
        "Title": title,
        "Description": description,
        "Class": category,
        "Subjects": " | ".join(subjects),
        "XML": os.path.basename(xml_path),
    }, "incluido"


def extract_dataset(xml_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    counters = Counter()
    records = []
    unclassified = []

    xml_files = []
    for current_dir, _, filenames in os.walk(xml_dir):
        for filename in filenames:
            if filename.lower().endswith(".xml"):
                xml_files.append(os.path.join(current_dir, filename))

    xml_files.sort()

    for index, xml_path in enumerate(xml_files, start=1):
        try:
            record, status = extract_record(xml_path)
            counters[status] += 1
            if record is not None and status == "incluido":
                records.append(record)
            elif record is not None and status == "tfg_sin_categoria":
                unclassified.append(record)
        except (ET.ParseError, OSError, UnicodeError) as exc:
            counters["error_xml"] += 1
            print(f"[AVISO] No se pudo leer {xml_path}: {exc}")

        if index % 1000 == 0:
            print(f"Procesados {index}/{len(xml_files)} XML...")

    columns = ["Title", "Description", "Class", "Subjects", "XML"]
    df = pd.DataFrame(records, columns=columns)
    csv_path = os.path.join(output_dir, "datos_clasificados.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    unclassified_df = pd.DataFrame(unclassified, columns=columns)
    unclassified_path = os.path.join(output_dir, "tfg_sin_categoria.csv")
    unclassified_df.to_csv(unclassified_path, index=False, encoding="utf-8-sig")

    counts = Counter(record["Class"] for record in records)
    summary_path = os.path.join(output_dir, "resumen_extraccion.txt")
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(f"XML encontrados: {len(xml_files)}\n")
        file.write(f"Registros TFG incluidos: {len(records)}\n")
        file.write(f"TFG sin categoría reconocida: {len(unclassified)}\n")
        file.write(f"Registros descartados por no ser TFG: {counters['no_es_tfg']}\n")
        file.write(f"TFG sin título: {counters['sin_titulo']}\n")
        file.write(f"TFG sin descripción: {counters['sin_descripcion']}\n")
        file.write(f"Errores de lectura XML: {counters['error_xml']}\n\n")
        file.write("Distribución de categorías:\n")
        for category in CATEGORIES:
            file.write(f"{category}: {counts.get(category, 0)}\n")

    print("\nExtracción terminada.")
    print(f"CSV de entrenamiento: {csv_path}")
    print(f"TFG sin categoría: {unclassified_path}")
    print(f"Resumen: {summary_path}")
    print("\nDistribución:")
    for category in CATEGORIES:
        print(f"  {category}: {counts.get(category, 0)}")

    return df


# ---------------------------------------------------------------------
# Preparación, entrenamiento y evaluación
# ---------------------------------------------------------------------

def create_model(vocab_size, sequence_length, number_of_classes):
    model = Sequential([
        Input(shape=(sequence_length,)),
        Embedding(input_dim=vocab_size, output_dim=50, mask_zero=True),
        LSTM(32),
        Dense(32, activation="relu"),
        Dense(number_of_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(df, output_dir):
    # Solo usamos las seis clases acordadas.
    df = df[df["Class"].isin(CATEGORIES)].copy()

    counts = df["Class"].value_counts()
    print("\nEjemplos disponibles para entrenar:")
    for category in CATEGORIES:
        print(f"  {category}: {int(counts.get(category, 0))}")

    # La división estratificada necesita al menos dos ejemplos por clase.
    usable_categories = [cat for cat in CATEGORIES if counts.get(cat, 0) >= 2]
    df = df[df["Class"].isin(usable_categories)].copy()

    if len(usable_categories) < 2:
        raise ValueError(
            "No hay suficientes ejemplos etiquetados en al menos dos categorías. "
            "Revisa tfg_sin_categoria.csv y las reglas DEGREE_RULES."
        )

    if len(df) < 10:
        raise ValueError("Hay menos de 10 TFG clasificados; no es suficiente para entrenar.")

    class_to_id = {category: index for index, category in enumerate(CATEGORIES)}
    y = df["Class"].map(class_to_id).to_numpy()
    texts = (df["Title"].fillna("") + ". " + df["Description"].fillna("")).to_numpy()

    # Si alguna clase solo tiene dos ejemplos, el test puede quedarse con uno.
    # El conjunto de prueba debe contener al menos un ejemplo de cada clase.
    test_count = max(len(usable_categories), int(np.ceil(len(df) * 0.2)))
    if len(df) - test_count < len(usable_categories):
        raise ValueError(
            "Hay muy pocos ejemplos para separar entrenamiento y prueba "
            "manteniendo todas las categorías. Necesitamos más TFG por categoría."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_count,
        random_state=SEED,
        stratify=y,
    )

    vectorizer = TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=MAX_SEQUENCE_LENGTH,
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.adapt(X_train)

    X_train_vec = vectorizer(X_train)
    X_test_vec = vectorizer(X_test)
    vocab_size = len(vectorizer.get_vocabulary())

    model = create_model(vocab_size, MAX_SEQUENCE_LENGTH, len(CATEGORIES))

    validation_fraction = 0.2
    history = model.fit(
        X_train_vec,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=validation_fraction,
        verbose=1,
        shuffle=True,
    )

    probabilities = model.predict(X_test_vec, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    present_ids = sorted(set(y_test) | set(y_pred))
    present_names = [CATEGORIES[i] for i in present_ids]

    precision_path = os.path.join(output_dir, "precision.txt")
    with open(precision_path, "w", encoding="utf-8") as file:
        file.write(f"Semilla aleatoria: {SEED}\n")
        file.write(f"Épocas: {EPOCHS}\n")
        file.write(f"Ejemplos entrenamiento: {len(X_train)}\n")
        file.write(f"Ejemplos prueba: {len(X_test)}\n")
        file.write(f"Tamaño vocabulario: {vocab_size}\n")
        file.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n")
        file.write(classification_report(
            y_test,
            y_pred,
            labels=present_ids,
            target_names=present_names,
            zero_division=0,
        ))

    matrix = confusion_matrix(y_test, y_pred, labels=present_ids)
    confusion_path = os.path.join(output_dir, "confusion.txt")
    with open(confusion_path, "w", encoding="utf-8") as file:
        file.write("Orden de categorías:\n")
        for index, name in zip(present_ids, present_names):
            file.write(f"{index}: {name}\n")
        file.write("\nMatriz de confusión (filas = reales; columnas = predichas):\n")
        file.write(np.array2string(matrix))

    # Curvas de pérdida de entrenamiento y validación.
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Entrenamiento")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Pérdida (loss)")
    plt.title("Evolución del error durante el entrenamiento")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error.jpg"), dpi=150)
    plt.close()

    model.save(os.path.join(output_dir, "modelo_lstm.keras"))
    with open(os.path.join(output_dir, "categorias.txt"), "w", encoding="utf-8") as file:
        for index, category in enumerate(CATEGORIES):
            file.write(f"{index}\t{category}\n")

    print("\nEntrenamiento y evaluación terminados.")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Resultados guardados en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extrae TFG de XML de Zaguán y entrena un clasificador LSTM."
    )
    parser.add_argument("-dir", required=True, help="Carpeta que contiene los XML.")
    parser.add_argument("-output", required=True, help="Carpeta donde guardar resultados.")
    parser.add_argument(
        "--solo-extraer",
        action="store_true",
        help="Extrae y resume los XML, pero no entrena todavía el modelo.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise NotADirectoryError(f"No existe la carpeta de XML: {args.dir}")

    set_random_seed(SEED)
    os.makedirs(args.output, exist_ok=True)

    df = extract_dataset(args.dir, args.output)
    if df.empty:
        raise ValueError(
            "No se ha podido clasificar ningún TFG con las reglas actuales. "
            "Revisa el resumen y tfg_sin_categoria.csv."
        )

    if args.solo_extraer:
        print("\nModo solo extracción: no se ha entrenado ningún modelo.")
        return

    train_and_evaluate(df, args.output)


if __name__ == "__main__":
    main()
