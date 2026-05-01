"""
Evaluacion de proyeccion NN para prediccion S2 en cohortes no vistas.

Para cada modelo TuckER entrenado (dataset x rdim):
  1. Extrae embeddings aprendidos de alumnos del train
  2. Construye features 4D (notas S1 normalizadas) para esos alumnos
  3. Entrena una NN que mapea notas_S1 (4D) -> embedding_aprendido (edim-D)
  4. Para cohortes OOT: proyecta notas de alumnos nuevos y scoring TuckER
"""
from __future__ import annotations

import sys, json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace
from model import TuckER
from load_data import Data

DF_BASE = REPO_ROOT / "data" / "dataframes_por_semestre"
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "resumen_s2_proyeccion_oot.csv"

CURSOS_S1 = ["MA1101", "MA1001", "FI1000", "BT1211"]
CURSOS_S2 = ["MA1002", "MA1102", "FI1100", "CC1002"]
CURSOS_PERMITIDOS = set(CURSOS_S1 + CURSOS_S2)

FAIL_GRADE = 1.0
APPROVED_TRANSFER = 4.0
EDIM = 200

# Cohortes S1->S2 disponibles
TODAS_COHORTES = [
    ("20191", "20192"),
    ("20201", "20202"),
    ("20211", "20212"),
    ("20221", "20222"),
    ("20231", "20232"),
]

# Qué cohortes usó cada dataset para entrenar
TRAIN_COHORTES = {
    "dataset_2021_hacia_s2_binario": [("20211", "20212")],
    "dataset_2020_2021_hacia_s2_binario": [("20201", "20202"), ("20211", "20212")],
    "dataset_2019_2020_2021_hacia_s2_binario": [("20191", "20192"), ("20201", "20202"), ("20211", "20212")],
}

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ═════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═════════════════════════════════════════════════════════════════════════

def _norm_cols(df):
    df = df.copy()
    df["ID"] = df["ID"].astype(str).str.strip().str.upper()
    df["CURSO"] = df["CURSO"].astype(str).str.strip().str.upper()
    df["ESTADO_CURSO"] = df["ESTADO_CURSO"].astype(str)
    return df


def _nota_efectiva(estado, nota_raw):
    if "Aprobado (T)" in str(estado) or "Aprobado(T)" in str(estado):
        return APPROVED_TRANSFER
    try:
        return float(str(nota_raw).replace(",", "."))
    except Exception:
        if "Aprobado" in str(estado):
            return APPROVED_TRANSFER
        return FAIL_GRADE


def _normalizar(n):
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return (FAIL_GRADE - 4.0) / 3.0
    return (float(n) - 4.0) / 3.0


def _determinar_relacion(estado, nota_raw):
    e = str(estado)
    if "Aprobado" in e and "Eliminado" not in e and "Reprobado" not in e:
        return "aprueba"
    if "Reprobado" in e or "Eliminado" in e:
        return "reprueba"
    try:
        nota = float(str(nota_raw).replace(",", "."))
        return "reprueba" if nota < 4.0 else "aprueba"
    except Exception:
        return None


def construir_notas_s1(ids, sem_s1):
    """Construye vector 4D de notas S1 normalizadas para un conjunto de alumnos."""
    ruta = DF_BASE / f"df_{sem_s1}.csv"
    if not ruta.exists():
        return {}
    df = _norm_cols(pd.read_csv(ruta, sep=";"))
    df = df[df["CURSO"].isin(CURSOS_S1)].copy()
    df["NOTA_EF"] = [_nota_efectiva(e, n) for e, n in zip(df["ESTADO_CURSO"], df["NOTA"])]
    pivot = df.pivot_table(index="ID", columns="CURSO", values="NOTA_EF", aggfunc="max")
    pivot = pivot.reindex(columns=CURSOS_S1).fillna(FAIL_GRADE)

    result = {}
    for alumno in ids:
        if alumno in pivot.index:
            row = pivot.loc[alumno]
            result[alumno] = np.array([_normalizar(row[c]) for c in CURSOS_S1], dtype=np.float32)
        else:
            result[alumno] = np.array([_normalizar(FAIL_GRADE)] * 4, dtype=np.float32)
    return result


# ═════════════════════════════════════════════════════════════════════════
# CARGAR MODELO TUCKER
# ═════════════════════════════════════════════════════════════════════════

def cargar_tucker(dataset_name, rdim):
    """Carga modelo y reconstruye vocabulario desde los datos."""
    folder = RESULTS_DIR / f"s2_{dataset_name}_rdim{rdim}"
    model_path = folder / f"s2_{dataset_name}_rdim{rdim}.pt"

    data_dir = f"data/{dataset_name}/"
    d = Data(data_dir=data_dir, reverse=True)

    model = TuckER(d, EDIM, rdim, input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    entity_idxs = {e: i for i, e in enumerate(d.entities)}
    relation_idxs = {r: i for i, r in enumerate(d.relations)}

    return model, d, entity_idxs, relation_idxs


# ═════════════════════════════════════════════════════════════════════════
# PREPARAR DATOS NN (notas -> embedding aprendido)
# ═════════════════════════════════════════════════════════════════════════

def preparar_datos_nn(model, d, entity_idxs, train_cohortes):
    """
    Para alumnos del train de TuckER:
      X = notas S1 normalizadas (4D)
      Y = embedding aprendido por TuckER (edim-D)
    """
    cursos_set = CURSOS_PERMITIDOS
    learned = model.E.weight.data

    # Identificar alumnos (entidades que no son cursos)
    alum_entities = [e for e in d.entities if e not in cursos_set
                     and e not in {c + "_REVERSE" for c in cursos_set}]

    # Construir notas S1 de todos los semestres de train
    notas_por_alumno = {}
    for sem_s1, _ in train_cohortes:
        notas = construir_notas_s1(alum_entities, sem_s1)
        notas_por_alumno.update(notas)

    X_list, Y_list = [], []
    for alumno in alum_entities:
        if alumno in notas_por_alumno and alumno in entity_idxs:
            idx = entity_idxs[alumno]
            X_list.append(notas_por_alumno[alumno])
            Y_list.append(learned[idx].numpy())

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
    print(f"  [Datos NN] Alumnos: {len(X_list)}, X shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y


# ═════════════════════════════════════════════════════════════════════════
# MODELOS NN
# ═════════════════════════════════════════════════════════════════════════

class MLPProyeccion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, output_dim),
        )
    def forward(self, x):
        return self.net(x)


def entrenar_nn(X, Y, edim, epochs=500, lr=1e-3, patience=100):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    modelo_nn = MLPProyeccion(4, edim)

    n = len(X)
    idx = np.random.permutation(n)
    n_tr = int(0.8 * n)
    n_va = int(0.15 * n)

    X_tr, Y_tr = X[idx[:n_tr]], Y[idx[:n_tr]]
    X_va, Y_va = X[idx[n_tr:n_tr + n_va]], Y[idx[n_tr:n_tr + n_va]]

    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=64, shuffle=True)
    opt = torch.optim.Adam(modelo_nn.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for ep in range(1, epochs + 1):
        modelo_nn.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(modelo_nn(xb), yb).backward()
            opt.step()

        modelo_nn.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(modelo_nn(X_va), Y_va))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in modelo_nn.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state:
        modelo_nn.load_state_dict(best_state)
    modelo_nn.eval()
    print(f"  [NN] val_loss: {best_val:.6f}, epochs: {ep}")
    return modelo_nn


# ═════════════════════════════════════════════════════════════════════════
# GENERAR TRIPLETAS OOT
# ═════════════════════════════════════════════════════════════════════════

def generar_tripletas_oot(sem_s1, sem_s2):
    """Genera tripletas de evaluación para una cohorte S1->S2."""
    ruta_s1 = DF_BASE / f"df_{sem_s1}.csv"
    ruta_s2 = DF_BASE / f"df_{sem_s2}.csv"
    if not ruta_s1.exists() or not ruta_s2.exists():
        return [], []

    df_s1 = _norm_cols(pd.read_csv(ruta_s1, sep=";"))
    df_s2 = _norm_cols(pd.read_csv(ruta_s2, sep=";"))

    # Alumnos con 4 cursos S1
    conteo = df_s1[df_s1["CURSO"].isin(CURSOS_S1)].groupby("ID")["CURSO"].nunique()
    ids_con_4 = set(conteo[conteo == 4].index)

    # Filtrar S2: al menos 1 curso permitido
    df_s2_filt = df_s2[df_s2["ID"].isin(ids_con_4) & df_s2["CURSO"].isin(CURSOS_PERMITIDOS)].copy()
    df_s2_filt["relacion"] = [_determinar_relacion(e, n)
                               for e, n in zip(df_s2_filt["ESTADO_CURSO"], df_s2_filt["NOTA"])]
    df_s2_filt = df_s2_filt.dropna(subset=["relacion"])

    tripletas = list(zip(df_s2_filt["ID"], df_s2_filt["relacion"], df_s2_filt["CURSO"]))
    ids_unicos = df_s2_filt["ID"].unique().tolist()
    return tripletas, ids_unicos


# ═════════════════════════════════════════════════════════════════════════
# EVALUACION
# ═════════════════════════════════════════════════════════════════════════

def evaluar_oot(modelo_nn, tucker_model, entity_idxs, relation_idxs, d,
                tripletas, emb_notas, rdim):
    rels_fwd = [r for r in d.relations if not r.endswith("_reverse")]
    rel_idxs_fwd = [relation_idxs[r] for r in rels_fwd]

    W_flat = tucker_model.W.data.view(rdim, EDIM * EDIM)
    W_mats = {}
    with torch.no_grad():
        for r_name, r_idx in zip(rels_fwd, rel_idxs_fwd):
            e_r = tucker_model.R.weight[r_idx]
            W_mats[r_name] = (e_r @ W_flat).view(EDIM, EDIM)

    resultados = []
    with torch.no_grad():
        for alumno, rel_true, curso in tripletas:
            if curso not in entity_idxs:
                continue

            x = torch.tensor(emb_notas[alumno], dtype=torch.float32).unsqueeze(0)
            e_h = modelo_nn(x).squeeze(0)
            e_h_bn = tucker_model.bn0(e_h.unsqueeze(0)).squeeze(0)

            t_idx = entity_idxs[curso]
            e_t = tucker_model.E.weight[t_idx]

            scores = {}
            for r_name in rels_fwd:
                W_r = W_mats[r_name]
                xr = e_h_bn @ W_r
                xr = tucker_model.bn1(xr.unsqueeze(0)).squeeze(0)
                scores[r_name] = float(xr @ e_t)

            orden = sorted(rels_fwd, key=lambda r: scores[r], reverse=True)
            rank = orden.index(rel_true) + 1 if rel_true in orden else len(rels_fwd) + 1
            pred_rel = orden[0]

            resultados.append({
                "alumno": alumno, "rel_true": rel_true, "rel_pred": pred_rel,
                "curso": curso, "rank": rank,
            })

    return resultados


def calcular_metricas(resultados):
    if not resultados:
        return {"n": 0}
    ranks = np.array([r["rank"] for r in resultados], dtype=float)
    rel_true = [r["rel_true"] for r in resultados]
    rel_pred = [r["rel_pred"] for r in resultados]
    n = len(ranks)

    TP = sum(1 for t, p in zip(rel_true, rel_pred) if t == "reprueba" and p == "reprueba")
    FP = sum(1 for t, p in zip(rel_true, rel_pred) if t == "aprueba" and p == "reprueba")
    FN = sum(1 for t, p in zip(rel_true, rel_pred) if t == "reprueba" and p == "aprueba")
    TN = sum(1 for t, p in zip(rel_true, rel_pred) if t == "aprueba" and p == "aprueba")

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "hits@1": float(np.mean(ranks == 1)),
        "mrr": float(np.mean(1.0 / ranks)),
        "acc": (TP + TN) / n,
        "prec_reprueba": prec,
        "recall_reprueba": rec,
        "f1_reprueba": f1,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN, "n": n,
    }


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    all_rows = []

    for dataset_name, train_cohs in TRAIN_COHORTES.items():
        oot_cohortes = [c for c in TODAS_COHORTES if c not in train_cohs]

        for rdim in range(1, 13):
            print(f"\n{'='*70}")
            print(f"  {dataset_name} | rdim={rdim}")
            print(f"{'='*70}")

            try:
                model, d, eidx, ridx = cargar_tucker(dataset_name, rdim)
            except Exception as e:
                print(f"  ERROR cargando modelo: {e}")
                continue

            X, Y = preparar_datos_nn(model, d, eidx, train_cohs)
            modelo_nn = entrenar_nn(X, Y, EDIM)

            for sem_s1, sem_s2 in oot_cohortes:
                tripletas, ids_oot = generar_tripletas_oot(sem_s1, sem_s2)
                if not tripletas:
                    print(f"  Cohorte {sem_s1}->{sem_s2}: sin datos")
                    continue

                emb_notas = construir_notas_s1(ids_oot, sem_s1)
                resultados = evaluar_oot(modelo_nn, model, eidx, ridx, d,
                                         tripletas, emb_notas, rdim)
                met = calcular_metricas(resultados)

                met["dataset"] = dataset_name
                met["rdim"] = rdim
                met["cohorte_oot"] = f"{sem_s1}->{sem_s2}"
                all_rows.append(met)

                print(f"  {sem_s1}->{sem_s2}: n={met['n']} | Acc={met.get('acc',0):.4f} | "
                      f"F1={met.get('f1_reprueba',0):.4f} | Hits@1={met.get('hits@1',0):.4f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResumen guardado en: {OUTPUT_CSV}")

    # Tabla resumen
    if len(df) > 0:
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", "{:.4f}".format)
        cols = ["dataset", "rdim", "cohorte_oot", "n", "acc", "f1_reprueba",
                "prec_reprueba", "recall_reprueba", "hits@1", "mrr"]
        print(f"\n{'='*100}")
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
