from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DF_BASE = REPO_ROOT / "data" / "dataframes_por_semestre"
OUTPUT_DATASET_DIR = REPO_ROOT / "data" / "dataset_2019_2020_2021_hacia_s3_binario"

# Cohortes de entrada y su semestre objetivo (tercer semestre academico).
PARES_SEMESTRES = [
    ("20191", "20201"),
    ("20201", "20211"),
    ("20211", "20221"),
]

CURSOS_PRIMER = {"MA1101", "MA1001", "FI1000", "BT1211"}
CURSOS_SEGUNDO = {"MA1002", "MA1102", "FI1100", "CC1002"}
CURSOS_TERCER = {"MA2001", "MA2601", "FI2001", "FI2003", "IQ2211"}
CURSOS_PERMITIDOS = CURSOS_PRIMER.union(CURSOS_SEGUNDO).union(CURSOS_TERCER)

SPLIT_RATIO = (0.8, 0.1, 0.1)
SEED = 42


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ID"] = out["ID"].astype(str).str.strip().str.upper()
    out["CURSO"] = out["CURSO"].astype(str).str.strip().str.upper()
    out["ESTADO_CURSO"] = out["ESTADO_CURSO"].astype(str)
    return out


def determinar_relacion(estado: object, nota_val: object) -> str | None:
    estado = str(estado)
    if "Aprobado" in estado:
        return "aprueba"
    if "Reprobado" in estado or "Eliminado" in estado:
        return "reprueba"
    try:
        val = float(str(nota_val).replace(",", "."))
        return "reprueba" if val < 4.0 else "aprueba"
    except Exception:
        return None


def leer_semestre(codigo_semestre: str) -> pd.DataFrame | None:
    ruta = DF_BASE / f"df_{codigo_semestre}.csv"
    if not ruta.exists():
        return None
    return normalizar_columnas(pd.read_csv(ruta, sep=";"))


def guardar(path: Path, data: list[str]) -> None:
    contenido = "\n".join(data)
    if contenido:
        contenido += "\n"
    path.write_text(contenido, encoding="utf-8")


def generar_tripletas() -> set[str]:
    print("\n--- FASE 1: GENERANDO TRIPLETAS BINARIAS PARA PREDICCION S3 ---")
    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    tripletas: list[str] = []
    ids_validos_global: set[str] = set()

    for sem_ant, sem_act in PARES_SEMESTRES:
        df_prev = leer_semestre(sem_ant)
        df_act = leer_semestre(sem_act)
        if df_prev is None or df_act is None:
            print(f"Saltando {sem_ant}->{sem_act}: falta df_{sem_ant}.csv o df_{sem_act}.csv")
            continue

        print(f"Procesando: {sem_ant} -> {sem_act}")

        # 1) Filtro S1: el alumno debe haber cursado los 4 cursos fundamentales de S1.
        df_fund_prev = df_prev[df_prev["CURSO"].isin(CURSOS_PRIMER)]
        conteo = df_fund_prev.groupby("ID")["CURSO"].nunique()
        alumnos_cumplen_s1 = set(conteo[conteo == len(CURSOS_PRIMER)].index)

        # 2) Filtro semestre objetivo: al menos 1 curso fundamental de S1/S2/S3.
        df_fund_act = df_act[df_act["CURSO"].isin(CURSOS_PERMITIDOS)]
        alumnos_cumplen_objetivo = set(df_fund_act["ID"].unique())

        # 3) Interseccion de elegibles.
        alumnos_validos = alumnos_cumplen_s1.intersection(alumnos_cumplen_objetivo)

        print(f"  Cumplen S1 (4 ramos): {len(alumnos_cumplen_s1)}")
        print(f"  Cumplen objetivo (>=1 ramo S1/S2/S3): {len(alumnos_cumplen_objetivo)}")
        print(f"  Interseccion valida: {len(alumnos_validos)}")

        ids_validos_global.update(alumnos_validos)

        # 4) Tripletas: solo cursos fundamentales de S1/S2/S3 en el semestre objetivo.
        df_filtrado = df_act[
            (df_act["ID"].isin(alumnos_validos))
            & (df_act["CURSO"].isin(CURSOS_PERMITIDOS))
        ]

        count_local = 0
        for _, row in df_filtrado.iterrows():
            rel = determinar_relacion(row["ESTADO_CURSO"], row["NOTA"])
            if rel:
                tripletas.append(f"{row['ID']}\t{rel}\t{row['CURSO']}")
                count_local += 1

        print(f"  Tripletas generadas: {count_local}")

    print(f"\nTotal tripletas sin split: {len(tripletas)}")

    # Archivo consolidado con todas las tripletas de entrenamiento solicitadas.
    guardar(OUTPUT_DATASET_DIR / "tripletas_entrenamiento_s3.txt", tripletas)

    # Split por alumno: evita fugas de informacion entre train/valid/test.
    alumnos_unicos = sorted({t.split("\t", 1)[0] for t in tripletas})
    rng.shuffle(alumnos_unicos)

    n_alumnos = len(alumnos_unicos)
    n_train_al = int(n_alumnos * SPLIT_RATIO[0])
    n_valid_al = int(n_alumnos * SPLIT_RATIO[1])

    ids_train = set(alumnos_unicos[:n_train_al])
    ids_valid = set(alumnos_unicos[n_train_al : n_train_al + n_valid_al])
    ids_test = set(alumnos_unicos[n_train_al + n_valid_al :])

    train_data = [t for t in tripletas if t.split("\t", 1)[0] in ids_train]
    valid_data = [t for t in tripletas if t.split("\t", 1)[0] in ids_valid]
    test_data = [t for t in tripletas if t.split("\t", 1)[0] in ids_test]

    rng.shuffle(train_data)
    rng.shuffle(valid_data)
    rng.shuffle(test_data)

    print(
        "Split por alumno -> "
        f"train: {len(ids_train)} alumnos / {len(train_data)} tripletas, "
        f"valid: {len(ids_valid)} alumnos / {len(valid_data)} tripletas, "
        f"test: {len(ids_test)} alumnos / {len(test_data)} tripletas"
    )

    guardar(OUTPUT_DATASET_DIR / "train.txt", train_data)
    guardar(OUTPUT_DATASET_DIR / "valid.txt", valid_data)
    guardar(OUTPUT_DATASET_DIR / "test.txt", test_data)
    guardar(OUTPUT_DATASET_DIR / "ids_train.txt", sorted(ids_train))
    guardar(OUTPUT_DATASET_DIR / "ids_valid.txt", sorted(ids_valid))
    guardar(OUTPUT_DATASET_DIR / "ids_test.txt", sorted(ids_test))

    # Balanceo igual al notebook: sobre-muestreo de reprueba en train.
    reprobados = [t for t in train_data if "\treprueba\t" in t]
    aprobados = [t for t in train_data if "\taprueba\t" in t]
    if len(reprobados) > 0 and len(aprobados) > len(reprobados):
        factor = len(aprobados) // len(reprobados)
        resto = len(aprobados) % len(reprobados)
        train_bal = aprobados + (reprobados * factor) + reprobados[:resto]
        rng.shuffle(train_bal)
        guardar(OUTPUT_DATASET_DIR / "train_balanceado.txt", train_bal)
        print(
            "Balanceo train: "
            f"{len(reprobados)} reprueba original -> {len(train_bal) - len(aprobados)} reprueba final"
        )
    else:
        guardar(OUTPUT_DATASET_DIR / "train_balanceado.txt", train_data)

    print(f"Dataset guardado en: {OUTPUT_DATASET_DIR}")
    return ids_validos_global


def main() -> None:
    ids_validos = generar_tripletas()
    print(f"IDs validos totales: {len(ids_validos)}")


if __name__ == "__main__":
    main()
