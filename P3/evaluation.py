"""Base de lectura de ficheros para la evaluación de un sistema de RI.

Este módulo solo prepara los datos de entrada. El cálculo de las medidas de
evaluación se añadirá posteriormente.
"""

import argparse
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Tuple


Qrels = Dict[str, Dict[str, int]]
Results = Dict[str, List[str]]
RecallPrecisionPoints = List[Tuple[float, float]]


def _read_fields(file_path: Path, expected_fields: int) -> List[Tuple[int, List[str]]]:
    """Lee líneas tabuladas y devuelve sus campos junto con su número de línea."""
    rows: List[Tuple[int, List[str]]] = []

    try:
        with file_path.open(encoding="utf-8") as input_file:
            for line_number, raw_line in enumerate(input_file, start=1):
                line = raw_line.rstrip("\r\n")
                if not line.strip():
                    continue

                fields = line.split("\t")
                if len(fields) != expected_fields or any(
                    not field.strip() for field in fields
                ):
                    raise ValueError(
                        f"{file_path}:{line_number}: se esperaban "
                        f"{expected_fields} valores separados por tabuladores"
                    )
                rows.append((line_number, [field.strip() for field in fields]))
    except OSError as exc:
        raise OSError(f"No se puede leer el fichero '{file_path}': {exc}") from exc

    return rows


def parse_qrels(file_path: Path) -> Qrels:
    """Parsea INFORMATION_NEED, DOCUMENT_ID y RELEVANCY del fichero qrels."""
    qrels: Qrels = {}

    for line_number, fields in _read_fields(file_path, expected_fields=3):
        information_need, document_id, relevance = fields
        if relevance not in {"0", "1"}:
            raise ValueError(
                f"{file_path}:{line_number}: la relevancia debe ser 0 o 1"
            )

        relevancias = qrels.setdefault(information_need, {})
        if document_id in relevancias:
            raise ValueError(
                f"{file_path}:{line_number}: esta relevancia aparece dos veces"
                f"{information_need}/{document_id}"
            )
        relevancias[document_id] = int(relevance)

    return qrels


def parse_results(file_path: Path) -> Results:
    """Parsea INFORMATION_NEED y DOCUMENT_ID respetando el orden de las líneas."""
    results: Results = {}

    for _, fields in _read_fields(file_path, expected_fields=2):
        information_need, document_id = fields
        results.setdefault(information_need, []).append(document_id)

    return results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara los ficheros qrels y de resultados para su evaluación."
    )
    parser.add_argument(
        "-qrels",
        required=True,
        type=Path,
        help="fichero con los juicios de relevancia",
    )
    parser.add_argument(
        "-results",
        required=True,
        type=Path,
        help="fichero con los resultados recuperados",
    )
    parser.add_argument(
        "-output",
        required=True,
        type=Path,
        help="fichero donde se escribirán las medidas de evaluación",
    )
    return parser.parse_args()


def calcular_recall_precision(ranking: List[str], relevancias: Dict[str, int], k: int) -> RecallPrecisionPoints:
    ranking_top_k = ranking[:k]
    total_relevantes = sum(relevancia == 1 for relevancia in relevancias.values())
    if total_relevantes == 0:
        return []

    puntos: RecallPrecisionPoints = []
    relevantes_recuperados = 0
    for posicion, documento in enumerate(ranking_top_k, start=1):
        if relevancias.get(documento, 0) == 1:
            relevantes_recuperados += 1
            precision = relevantes_recuperados / posicion
            recall = relevantes_recuperados / total_relevantes
            puntos.append((recall, precision))

    return puntos


def calcular_recall_precision_interpolado(puntos: RecallPrecisionPoints,) -> RecallPrecisionPoints:
    puntos_interpolados: RecallPrecisionPoints = []
    for nivel_recall in (indice / 10 for indice in range(11)):
        precisiones = [
            precision
            for recall, precision in puntos
            if recall >= nivel_recall
        ]
        precision = max(precisiones, default=0) #default 0
        puntos_interpolados.append((nivel_recall, precision))
    return puntos_interpolados


#se ha pedido a un llm esta funcion para tener la opcion de imprimir por pantalla o por fichero por motivos de debug
def generar_output(qrels, results, outputFile, k=45):
    """Genera el informe y guarda toda la salida en ``outputFile``."""
    try:
        with outputFile.open("w", encoding="utf-8") as output:
            with redirect_stdout(output):
                _generar_output(qrels, results, k)
    except OSError as exc:
        raise OSError(
            f"No se puede escribir el fichero '{outputFile}': {exc}"
        ) from exc


def _generar_output(qrels, results, k=45):
    precisionGlobal = 0
    recallGlobal = 0
    precisionAt10Global = 0
    medidaMAP = 0
    interpolated_precision_recallGlobal: RecallPrecisionPoints = [
        (indice / 10, 0.0) for indice in range(11)
    ]
    
    n_necesidades = len(qrels)
    for information_need, relevancias in qrels.items():
        print(f"INFORMATION_NEED {information_need}")

        documentos_recuperados = set(results.get(information_need, []))
        ranking = results.get(information_need, [])
        documentos_juzgados = set(relevancias)
        documentos = documentos_juzgados | documentos_recuperados

        n_true_positive = sum(
            relevancias.get(documento, 0) == 1
            for documento in documentos_recuperados
        )
        n_false_positive = sum(
            relevancias.get(documento, 0) == 0
            for documento in documentos_recuperados
        )
        n_false_negative = sum(
            relevancia == 1 and documento not in documentos_recuperados
            for documento, relevancia in relevancias.items()
        )
        
        precision = n_true_positive / (n_true_positive + n_false_positive)
        precisionGlobal += precision
        
        recall = n_true_positive / (n_true_positive + n_false_negative)
        recallGlobal += recall
        
        f1score = 2 * precision * recall / (precision + recall)  # B=1
    
        precisionAt10 = sum(  # precision con los primeros 10 documentos
            relevancias.get(documento, 0) == 1
            for documento in ranking[:10]
        ) / 10
        precisionAt10Global+=precisionAt10
        
        ranking_top_k = ranking[:k]
        precisiones_en_relevantes = [
            sum(
                relevancias.get(documento, 0) == 1
                for documento in ranking_top_k[:posicion]
            ) / posicion
            for posicion, documento in enumerate(ranking_top_k, start=1)
            if relevancias.get(documento, 0) == 1
        ]
        precisionPromedio = (
            sum(precisiones_en_relevantes) / len(precisiones_en_relevantes)
            if precisiones_en_relevantes
            else 0
        )
        medidaMAP+=precisionPromedio

        print(f"precision {precision:.3f}")
        print(f"recall {recall:.3f}")
        print(f"F1 {f1score:.3f}")
        print(f"prec@10 {precisionAt10:.3f}")
        print(f"average_precision {precisionPromedio:.3f}")

        precision_recall = calcular_recall_precision(ranking, relevancias, k)
        interpolated_precision_recall = (
            calcular_recall_precision_interpolado(precision_recall)
        )
        for i in range(11):
            recall_point, precision_point = interpolated_precision_recallGlobal[i]
            interpolated_precision_recallGlobal[i] = (
                recall_point,
                precision_point + interpolated_precision_recall[i][1],
            )

        print("recall_precision")
        for recall_point, precision_point in precision_recall:
            print(f"{recall_point:.3f} {precision_point:.3f}")

        print("interpolated_recall_precision")
        for recall_point, precision_point in interpolated_precision_recall:
            print(f"{recall_point:.3f} {precision_point:.3f}")
        print("") #un \n

    print(f"TOTAL")
    precisionGlobal = precisionGlobal / n_necesidades
    recallGlobal = recallGlobal / n_necesidades
    f1scoreGlobal = 2 * precisionGlobal * recallGlobal / (
        precisionGlobal + recallGlobal
    )  # no es media de F1, es f1 calculado con medias globales
    precisionAt10Global = precisionAt10Global / n_necesidades
    medidaMAP = medidaMAP / n_necesidades
    interpolated_precision_recallGlobal = [
        (recall_point, precision_point / n_necesidades)
        for recall_point, precision_point in interpolated_precision_recallGlobal
    ]
    print(f"precision {precisionGlobal:.3f}")
    print(f"recall {recallGlobal:.3f}")
    print(f"F1 {f1scoreGlobal:.3f}")
    print(f"prec@10 {precisionAt10Global:.3f}")
    print(f"MAP {medidaMAP:.3f}")
    print("interpolated_recall_precision")
    for recall_point, precision_point in interpolated_precision_recallGlobal:
        print(f"{recall_point:.3f} {precision_point:.3f}")



    
    

def main() -> None:
    args = parse_arguments()
    qrels = parse_qrels(args.qrels)
    results = parse_results(args.results)

    generar_output(qrels, results, args.output)


if __name__ == "__main__":
    main()