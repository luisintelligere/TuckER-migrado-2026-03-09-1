"""Genera embeddings iniciales 4D (notas S1) y vocabulario JSON para warm-start de TuckER."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from load_data import Data

DF_BASE = REPO_ROOT / "data" / "dataframes_por_semestre"

CURSOS_PRIMER = ["BT1211", "FI1000", "MA1001", "MA1101"]  # sorted para consistencia 4D

# Mapeo: 1 cohorte usa solo 20211, 2 cohortes 20201+20211, 3 cohortes 20191+20201+20211
# 4: dataset aumentado (2021 completo + reprueba de 2019/2020 con tripletas completas)
SEMESTRES_S1_POR_COHORTE = {
    1: ["20211"],
    2: ["20201", "20211"],
    3: ["20191", "20201", "20211"],
    4: ["20191", "20201", "20211"],  # augmentado: notas S1 de las 3 cohortes
}

DATASETS = {
    1: "dataset_2021_hacia_s2_binario",
    2: "dataset_2020_2021_hacia_s2_binario",
    3: "dataset_2019_2020_2021_hacia_s2_binario",
    4: "dataset_augmentado_1920reprueba_2021_s2",
}


def normalizar_nota(n: float) -> float:
    """Normaliza nota al rango [-1, 1]. 4.0 -> 0, 7.0 -> 1, 1.0 -> -1."""
    if pd.isna(n):
        return -1.0
    return (n - 4.0) / 3.0


def construir_tabla_notas(semestres_s1: list[str]) -> pd.DataFrame:
    dfs = []
    for sem in semestres_s1:
        ruta = DF_BASE / f"df_{sem}.csv"
        if ruta.exists():
            df = pd.read_csv(ruta, sep=";")
            df["ID"] = df["ID"].astype(str).str.strip().str.upper()
            df["CURSO"] = df["CURSO"].astype(str).str.strip().str.upper()
            df["NOTA"] = pd.to_numeric(df["NOTA"], errors="coerce")
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    df_total = pd.concat(dfs, ignore_index=True)
    df_fund = df_total[df_total["CURSO"].isin(CURSOS_PRIMER)]
    pivot = df_fund.pivot_table(index="ID", columns="CURSO", values="NOTA", aggfunc="first")
    return pivot.reindex(columns=CURSOS_PRIMER)


def generar_embeddings(n_cohortes: int) -> None:
    dataset_name = DATASETS[n_cohortes]
    data_dir = REPO_ROOT / "data" / dataset_name
    output_dir = data_dir / "embeddings_4d"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Generando embeddings 4D para {dataset_name} ---")

    # Cargar vocabulario del dataset
    d = Data(data_dir=str(data_dir) + "/", reverse=True)
    entities = d.entities
    relations = d.relations

    print(f"  Entidades: {len(entities)}, Relaciones: {len(relations)}")

    # Construir tabla de notas S1
    semestres = SEMESTRES_S1_POR_COHORTE[n_cohortes]
    tabla_notas = construir_tabla_notas(semestres)
    print(f"  Alumnos con notas S1: {len(tabla_notas)}")

    # Crear tensor de embeddings (n_entities x 4)
    entity_idxs = {e: i for i, e in enumerate(entities)}
    emb = torch.zeros(len(entities), len(CURSOS_PRIMER))

    count_inicializados = 0
    for entity, idx in entity_idxs.items():
        if entity in tabla_notas.index:
            row = tabla_notas.loc[entity]
            vec = [normalizar_nota(row[c]) for c in CURSOS_PRIMER]
            emb[idx] = torch.tensor(vec, dtype=torch.float32)
            count_inicializados += 1

    print(f"  Embeddings inicializados con notas: {count_inicializados}/{len(entities)}")

    # Guardar
    emb_path = output_dir / "embeddings_inicializados_4d.pt"
    vocab_path = output_dir / "vocabulario_4d.json"

    torch.save(emb, emb_path)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"entities": entities, "relations": relations}, f, indent=2)

    print(f"  Guardado: {emb_path}")
    print(f"  Guardado: {vocab_path}")


def main() -> None:
    if len(sys.argv) > 1:
        valores = [int(x) for x in sys.argv[1:]]
    else:
        valores = [1, 2, 3]
    for n in valores:
        generar_embeddings(n)


if __name__ == "__main__":
    main()
