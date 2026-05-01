"""Genera dataset aumentado para prediccion S1->S2.

Estrategia:
  - Cohorte 2021 (20211->20212): TODOS los alumnos elegibles, TODAS sus tripletas S2.
  - Cohortes 2019/2020 (20191->20192, 20201->20202): solo alumnos con >=1 reprueba en S2,
    pero incluyendo TODAS sus tripletas S2 (aprueba Y reprueba) para evitar falsos negativos.

Un alumno es elegible si:
  1. Curso los 4 ramos fundamentales de S1 (BT1211, FI1000, MA1001, MA1101).
  2. Aparece en al menos 1 curso permitido en el semestre S2.

Dataset de salida: dataset_augmentado_1920reprueba_2021_s2
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DF_BASE = REPO_ROOT / "data" / "dataframes_por_semestre"

CURSOS_PRIMER = {"MA1101", "MA1001", "FI1000", "BT1211"}
CURSOS_SEGUNDO = {"MA1002", "MA1102", "FI1100", "CC1002"}
CURSOS_PERMITIDOS = CURSOS_PRIMER.union(CURSOS_SEGUNDO)

DATASET_NAME = "dataset_augmentado_1920reprueba_2021_s2"

# Cohortes: (sem_s1, sem_s2, modo)
# "all"              -> todos los alumnos elegibles, todas sus tripletas
# "reprueba_students"-> solo alumnos con >=1 reprueba, pero TODAS sus tripletas
COHORTES = [
    ("20191", "20192", "reprueba_students"),
    ("20201", "20202", "reprueba_students"),
    ("20211", "20212", "all"),
]

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
    if "Aprobado" in estado and "Reprobado" not in estado and "Eliminado" not in estado:
        return "aprueba"
    if "Reprobado" in estado or "Eliminado" in estado:
        return "reprueba"
    try:
        val = float(str(nota_val).replace(",", "."))
        return "reprueba" if val < 4.0 else "aprueba"
    except Exception:
        return None


def leer_semestre(codigo: str) -> pd.DataFrame | None:
    ruta = DF_BASE / f"df_{codigo}.csv"
    if not ruta.exists():
        print(f"  Archivo no encontrado: {ruta}")
        return None
    return normalizar_columnas(pd.read_csv(ruta, sep=";"))


def guardar(path: Path, data: list[str]) -> None:
    contenido = "\n".join(data)
    if contenido:
        contenido += "\n"
    path.write_text(contenido, encoding="utf-8")


def main() -> None:
    output_dir = REPO_ROOT / "data" / DATASET_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    tripletas: list[str] = []
    resumen_cohortes: list[dict] = []

    for sem_s1, sem_s2, modo in COHORTES:
        print(f"\n--- Procesando {sem_s1}->{sem_s2} (modo: {modo}) ---")

        df_prev = leer_semestre(sem_s1)
        df_act = leer_semestre(sem_s2)
        if df_prev is None or df_act is None:
            print(f"  Saltando: falta df_{sem_s1}.csv o df_{sem_s2}.csv")
            continue

        # 1) Alumnos con los 4 ramos fundamentales de S1
        df_fund_prev = df_prev[df_prev["CURSO"].isin(CURSOS_PRIMER)]
        conteo = df_fund_prev.groupby("ID")["CURSO"].nunique()
        alumnos_4s1 = set(conteo[conteo == len(CURSOS_PRIMER)].index)

        # 2) Alumnos con al menos 1 curso permitido en S2
        df_fund_act = df_act[df_act["CURSO"].isin(CURSOS_PERMITIDOS)]
        alumnos_en_s2 = set(df_fund_act["ID"].unique())

        # 3) Interseccion elegible (cumple ambas condiciones)
        alumnos_elegibles = alumnos_4s1.intersection(alumnos_en_s2)
        print(f"  4 ramos S1: {len(alumnos_4s1)} | En S2: {len(alumnos_en_s2)} | "
              f"Elegibles: {len(alumnos_elegibles)}")

        # 4) Seleccion de alumnos segun modo
        if modo == "reprueba_students":
            # Calcular relaciones para todos los elegibles en S2
            df_elig = df_fund_act[df_fund_act["ID"].isin(alumnos_elegibles)].copy()
            df_elig["relacion"] = [
                determinar_relacion(e, n)
                for e, n in zip(df_elig["ESTADO_CURSO"], df_elig["NOTA"])
            ]
            # Seleccionar solo alumnos con al menos 1 reprueba
            alumnos_con_reprueba = set(
                df_elig[df_elig["relacion"] == "reprueba"]["ID"].unique()
            )
            alumnos_seleccionados = alumnos_elegibles.intersection(alumnos_con_reprueba)
            print(f"  Alumnos elegibles con >=1 reprueba: {len(alumnos_con_reprueba)}")
            print(f"  Seleccionados: {len(alumnos_seleccionados)}")
        else:
            # "all": todos los elegibles
            alumnos_seleccionados = alumnos_elegibles
            print(f"  Seleccionados (todos): {len(alumnos_seleccionados)}")

        # 5) Generar TODAS las tripletas de esos alumnos (sin filtrar por relacion)
        # Esto es clave: incluso para modo "reprueba_students" se incluyen sus apruebas
        # para evitar falsos negativos en el KG.
        df_filtrado = df_act[
            df_act["ID"].isin(alumnos_seleccionados)
            & df_act["CURSO"].isin(CURSOS_PERMITIDOS)
        ].copy()

        count_aprueba = 0
        count_reprueba = 0
        for _, row in df_filtrado.iterrows():
            rel = determinar_relacion(row["ESTADO_CURSO"], row["NOTA"])
            if rel:
                tripletas.append(f"{row['ID']}\t{rel}\t{row['CURSO']}")
                if rel == "aprueba":
                    count_aprueba += 1
                else:
                    count_reprueba += 1

        resumen_cohortes.append({
            "cohorte": f"{sem_s1}->{sem_s2}",
            "modo": modo,
            "alumnos_seleccionados": len(alumnos_seleccionados),
            "tripletas_aprueba": count_aprueba,
            "tripletas_reprueba": count_reprueba,
        })
        print(f"  Tripletas: {count_aprueba} aprueba, {count_reprueba} reprueba "
              f"(total: {count_aprueba + count_reprueba})")

    print(f"\n{'='*60}")
    print(f"Total tripletas sin split: {len(tripletas)}")

    total_aprueba = sum(t["tripletas_aprueba"] for t in resumen_cohortes)
    total_reprueba = sum(t["tripletas_reprueba"] for t in resumen_cohortes)
    print(f"Global: {total_aprueba} aprueba, {total_reprueba} reprueba")

    # Guardar consolidado
    guardar(output_dir / "tripletas_entrenamiento_s2.txt", tripletas)

    # Split por alumno: todas las tripletas de un mismo alumno van al mismo set
    alumnos_unicos = sorted({t.split("\t", 1)[0] for t in tripletas})
    rng.shuffle(alumnos_unicos)

    n = len(alumnos_unicos)
    n_tr = int(n * SPLIT_RATIO[0])
    n_va = int(n * SPLIT_RATIO[1])

    ids_train = set(alumnos_unicos[:n_tr])
    ids_valid = set(alumnos_unicos[n_tr : n_tr + n_va])
    ids_test = set(alumnos_unicos[n_tr + n_va :])

    train_data = [t for t in tripletas if t.split("\t", 1)[0] in ids_train]
    valid_data = [t for t in tripletas if t.split("\t", 1)[0] in ids_valid]
    test_data = [t for t in tripletas if t.split("\t", 1)[0] in ids_test]

    rng.shuffle(train_data)
    rng.shuffle(valid_data)
    rng.shuffle(test_data)

    print(
        f"\nSplit por alumno ->"
        f" train: {len(ids_train)} alumnos / {len(train_data)} tripletas"
        f" | valid: {len(ids_valid)} alumnos / {len(valid_data)} tripletas"
        f" | test:  {len(ids_test)} alumnos / {len(test_data)} tripletas"
    )

    guardar(output_dir / "train.txt", train_data)
    guardar(output_dir / "valid.txt", valid_data)
    guardar(output_dir / "test.txt", test_data)
    guardar(output_dir / "ids_train.txt", sorted(ids_train))
    guardar(output_dir / "ids_valid.txt", sorted(ids_valid))
    guardar(output_dir / "ids_test.txt", sorted(ids_test))

    # Balanceo en train: sobre-muestreo de reprueba
    reprobados_tr = [t for t in train_data if "\treprueba\t" in t]
    aprobados_tr = [t for t in train_data if "\taprueba\t" in t]
    print(f"\nTrain original: {len(aprobados_tr)} aprueba, {len(reprobados_tr)} reprueba")

    if len(reprobados_tr) > 0 and len(aprobados_tr) > len(reprobados_tr):
        factor = len(aprobados_tr) // len(reprobados_tr)
        resto = len(aprobados_tr) % len(reprobados_tr)
        train_bal = aprobados_tr + (reprobados_tr * factor) + reprobados_tr[:resto]
        rng.shuffle(train_bal)
        guardar(output_dir / "train_balanceado.txt", train_bal)
        print(
            f"Balanceo: reprueba {len(reprobados_tr)} -> "
            f"{len(train_bal) - len(aprobados_tr)} (factor x{factor})"
        )
    else:
        guardar(output_dir / "train_balanceado.txt", train_data)
        print("Balanceo: no necesario (clases ya equilibradas o sin reprueba)")

    print(f"\nDataset guardado en: {output_dir}")
    print("\nResumen por cohorte:")
    for r in resumen_cohortes:
        print(
            f"  {r['cohorte']} [{r['modo']}]: {r['alumnos_seleccionados']} alumnos, "
            f"{r['tripletas_aprueba']} aprueba, {r['tripletas_reprueba']} reprueba"
        )


if __name__ == "__main__":
    main()
