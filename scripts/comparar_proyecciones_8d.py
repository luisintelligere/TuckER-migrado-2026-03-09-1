# -*- coding: utf-8 -*-
"""
Experimento comparativo: metodos de proyeccion R^8 -> R^8
para obtener embeddings de alumnos no vistos por Tucker.

Metodos evaluados:
  1. Identidad (usar notas directamente, baseline)
  2. Lineal (W*x + b)
  3. MLP original del informe (64-128, ReLU, Dropout)
  4. MLP mas profundo (64-128-64)
  5. Residual (x + MLP(x))

Cada uno se entrena con alumnos del train de Tucker y se evalua
con el pipeline completo sobre la cohorte 2022->2023.
"""
import sys, os, json
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

# ── Rutas ────────────────────────────────────────────────────────────────
DF_BASE      = REPO_ROOT / "data" / "dataframes_por_semestre"
EMB_INIT     = (REPO_ROOT / "notebooks" / "Experimento_warm_start"
                / "embeddings_8d_binario_s3" / "embeddings_inicializados_binarios_8d.pt")
MODEL_PATH   = REPO_ROOT / "results" / "prueba_s3_8d_rdim11" / "best_model.pt"
VOCAB_PATH   = (REPO_ROOT / "notebooks" / "Experimento_warm_start"
                / "embeddings_8d_binario_s3" / "vocabulario_binario_8d.json")
OUTPUT_DIR   = REPO_ROOT / "results" / "comparacion_proyecciones_8d"

# Cohorte OOT
SEM_S1_OOT  = "20221"
SEM_S2_OOT  = "20222"
SEM_OBJ_OOT = "20231"

CURSOS_S1 = ["MA1101", "MA1001", "FI1000", "BT1211"]
CURSOS_S2 = ["MA1002", "MA1102", "FI1100", "CC1002"]
CURSOS_S3 = ["MA2001", "MA2601", "FI2001", "FI2003", "IQ2211"]
CURSOS_8D = CURSOS_S1 + CURSOS_S2
CURSOS_PERMITIDOS = set(CURSOS_S1 + CURSOS_S2 + CURSOS_S3)

FAIL_GRADE        = 1.0
APPROVED_TRANSFER = 4.0
EDIM, RDIM        = 8, 11

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ═════════════════════════════════════════════════════════════════════════
# DATOS AUXILIARES
# ═════════════════════════════════════════════════════════════════════════

def _norm_cols(df):
    df = df.copy()
    df["ID"]           = df["ID"].astype(str).str.strip().str.upper()
    df["CURSO"]        = df["CURSO"].astype(str).str.strip().str.upper()
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

def _construir_pivot(sem, cursos):
    ruta = DF_BASE / f"df_{sem}.csv"
    if not ruta.exists():
        return pd.DataFrame(columns=cursos)
    df = _norm_cols(pd.read_csv(ruta, sep=";"))
    df = df[df["CURSO"].isin(cursos)].copy()
    df["NOTA_EF"] = [_nota_efectiva(e, n) for e, n in zip(df["ESTADO_CURSO"], df["NOTA"])]
    pivot = df.pivot_table(index="ID", columns="CURSO", values="NOTA_EF", aggfunc="max")
    return pivot.reindex(columns=cursos)

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

def construir_emb_8d(ids, sem_s1, sem_s2):
    p1 = _construir_pivot(sem_s1, CURSOS_S1).fillna(FAIL_GRADE)
    p2 = _construir_pivot(sem_s2, CURSOS_S2).fillna(FAIL_GRADE)
    p1 = p1[p1.index.isin(ids)]
    p2 = p2[p2.index.isin(ids)]
    tabla = p1.join(p2, how="outer").fillna(FAIL_GRADE).reindex(columns=CURSOS_8D)
    result = {}
    for alumno in ids:
        if alumno in tabla.index:
            row = tabla.loc[alumno]
            result[alumno] = np.array([_normalizar(row[c]) for c in CURSOS_8D], dtype=np.float32)
        else:
            result[alumno] = np.array([_normalizar(FAIL_GRADE)] * 8, dtype=np.float32)
    return result


# ═════════════════════════════════════════════════════════════════════════
# CARGAR MODELO Y VOCAB
# ═════════════════════════════════════════════════════════════════════════

def cargar_tucker():
    with open(VOCAB_PATH, encoding="utf-8") as f:
        vocab = json.load(f)
    d = SimpleNamespace()
    d.entities      = vocab["entities"]
    d.relations     = vocab["relations"]
    d.entity_idxs   = {e: i for i, e in enumerate(d.entities)}
    d.relation_idxs = {r: i for i, r in enumerate(d.relations)}
    model = TuckER(d, EDIM, RDIM, input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, d


# ═════════════════════════════════════════════════════════════════════════
# PREPARAR PARES (input, target) PARA ENTRENAR LA NN
# ═════════════════════════════════════════════════════════════════════════

def preparar_datos_nn(model, d):
    """
    Retorna:
      X_train, Y_train: tensores con notas 8D -> embedding aprendido
      Solo para alumnos del entrenamiento Tucker (no cursos).
    """
    init_tensor = torch.load(EMB_INIT, map_location="cpu")
    learned     = model.E.weight.data

    cursos_set = set(CURSOS_S1 + CURSOS_S2 + CURSOS_S3)
    alum_idxs = [i for i, e in enumerate(d.entities) if e not in cursos_set]

    X = init_tensor[alum_idxs]    # (N, 8) notas normalizadas
    Y = learned[alum_idxs]        # (N, 8) embedding aprendido

    print(f"\n[Datos NN] Alumnos de entrenamiento: {len(alum_idxs)}")

    # Analisis de drift
    diff = Y - X
    print(f"  Norm L2 diff promedio:  {float(diff.norm(dim=1).mean()):.4f}")
    print(f"  Norm L2 init promedio:  {float(X.norm(dim=1).mean()):.4f}")
    print(f"  Norm L2 learned prom:   {float(Y.norm(dim=1).mean()):.4f}")
    print(f"  Ratio drift/init:       {float(diff.norm(dim=1).mean() / X.norm(dim=1).mean()):.4f}")
    for dim in range(8):
        r = float(np.corrcoef(X[:, dim].numpy(), Y[:, dim].numpy())[0, 1])
        print(f"    Dim {dim}: pearson r = {r:.4f}")

    return X, Y


# ═════════════════════════════════════════════════════════════════════════
# DEFINIR MODELOS
# ═════════════════════════════════════════════════════════════════════════

class IdentidadModel(nn.Module):
    """Baseline: no transforma nada."""
    def forward(self, x):
        return x

class LinealModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
    def forward(self, x):
        return self.fc(x)

class MLPOriginal(nn.Module):
    """Arquitectura del informe: 8 -> 64 -> 128 -> 8"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 8),
        )
    def forward(self, x):
        return self.net(x)

class MLPProfundo(nn.Module):
    """Mas capas: 8 -> 64 -> 128 -> 64 -> 8"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 8),
        )
    def forward(self, x):
        return self.net(x)

class ResidualModel(nn.Module):
    """x + MLP(x) — la NN aprende solo el DELTA."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 8),
        )
    def forward(self, x):
        return x + self.net(x)


MODELOS = {
    "1_identidad":     lambda: IdentidadModel(),
    "2_lineal":        lambda: LinealModel(),
    "3_mlp_64_128":    lambda: MLPOriginal(),
    "4_mlp_64_128_64": lambda: MLPProfundo(),
    "5_residual":      lambda: ResidualModel(),
}


# ═════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO NN
# ═════════════════════════════════════════════════════════════════════════

def entrenar_nn(modelo_nn, X, Y, epochs=500, lr=1e-3, patience=200):
    """Entrena la NN con split 80/15/5 y early stopping."""
    n = len(X)
    idx = np.random.permutation(n)
    n_tr = int(0.8 * n)
    n_va = int(0.15 * n)

    X_tr, Y_tr = X[idx[:n_tr]], Y[idx[:n_tr]]
    X_va, Y_va = X[idx[n_tr:n_tr+n_va]], Y[idx[n_tr:n_tr+n_va]]
    X_te, Y_te = X[idx[n_tr+n_va:]], Y[idx[n_tr+n_va:]]

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

    with torch.no_grad():
        test_loss = float(loss_fn(modelo_nn(X_te), Y_te))

    return best_val, test_loss, ep


# ═════════════════════════════════════════════════════════════════════════
# EVALUACION PIPELINE COMPLETO (cohorte 2022 -> 2023)
# ═════════════════════════════════════════════════════════════════════════

def generar_tripletas_oot():
    df_s1 = _norm_cols(pd.read_csv(DF_BASE / f"df_{SEM_S1_OOT}.csv", sep=";"))
    alumnos_con_4 = (
        df_s1[df_s1["CURSO"].isin(CURSOS_S1)].groupby("ID")["CURSO"].nunique()
    )
    ids_validos = set(alumnos_con_4[alumnos_con_4 == 4].index)

    df_obj = _norm_cols(pd.read_csv(DF_BASE / f"df_{SEM_OBJ_OOT}.csv", sep=";"))
    df_obj = df_obj[
        df_obj["ID"].isin(ids_validos) & df_obj["CURSO"].isin(CURSOS_PERMITIDOS)
    ].copy()
    df_obj["relacion"] = [_determinar_relacion(e, n)
                          for e, n in zip(df_obj["ESTADO_CURSO"], df_obj["NOTA"])]
    df_obj = df_obj.dropna(subset=["relacion"])
    tripletas = list(zip(df_obj["ID"], df_obj["relacion"], df_obj["CURSO"]))
    ids_unicos = df_obj["ID"].unique().tolist()
    return tripletas, ids_unicos


def evaluar_con_proyeccion(modelo_nn, tucker_model, d, tripletas, emb_notas_oot):
    """
    Para cada tripleta:
      1. Construye embedding 8D del alumno OOT
      2. Lo proyecta con modelo_nn
      3. Scoring Tucker con el embedding proyectado
      4. Rankea relaciones
    """
    rels_fwd = [r for r in d.relations if not r.endswith("_reverse")]
    rel_idxs_fwd = [d.relation_idxs[r] for r in rels_fwd]

    W_flat = tucker_model.W.data.view(RDIM, EDIM * EDIM)
    W_mats = {}
    with torch.no_grad():
        for r_name, r_idx in zip(rels_fwd, rel_idxs_fwd):
            e_r = tucker_model.R.weight[r_idx]
            W_mats[r_name] = (e_r @ W_flat).view(EDIM, EDIM)

    resultados = []
    with torch.no_grad():
        for alumno, rel_true, curso in tripletas:
            if curso not in d.entity_idxs:
                continue

            # Embedding de notas -> proyectado
            x = torch.tensor(emb_notas_oot[alumno], dtype=torch.float32).unsqueeze(0)
            e_h = modelo_nn(x).squeeze(0)

            # Aplicar bn0
            e_h_bn = tucker_model.bn0(e_h.unsqueeze(0)).squeeze(0)

            t_idx = d.entity_idxs[curso]
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
    ranks    = np.array([r["rank"] for r in resultados], dtype=float)
    rel_true = [r["rel_true"] for r in resultados]
    rel_pred = [r["rel_pred"] for r in resultados]
    n = len(ranks)

    hits1 = float(np.mean(ranks == 1))
    hits2 = float(np.mean(ranks <= 2))
    mr    = float(np.mean(ranks))
    mrr   = float(np.mean(1.0 / ranks))

    TP = sum(1 for t, p in zip(rel_true, rel_pred) if t == "reprueba" and p == "reprueba")
    FP = sum(1 for t, p in zip(rel_true, rel_pred) if t == "aprueba" and p == "reprueba")
    FN = sum(1 for t, p in zip(rel_true, rel_pred) if t == "reprueba" and p == "aprueba")
    TN = sum(1 for t, p in zip(rel_true, rel_pred) if t == "aprueba" and p == "aprueba")

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (TP + TN) / n if n > 0 else 0.0

    return {
        "hits@1": hits1, "hits@2": hits2, "mr": mr, "mrr": mrr,
        "acc": acc, "prec_reprueba": prec, "recall_reprueba": rec, "f1_reprueba": f1,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN, "n": n,
    }


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("COMPARACION DE METODOS DE PROYECCION R^8 -> R^8")
    print("=" * 65)

    tucker_model, d = cargar_tucker()
    X, Y = preparar_datos_nn(tucker_model, d)

    # Datos OOT
    tripletas_oot, ids_oot = generar_tripletas_oot()
    emb_oot = construir_emb_8d(ids_oot, SEM_S1_OOT, SEM_S2_OOT)
    print(f"\n[OOT] Tripletas: {len(tripletas_oot)} | Alumnos: {len(ids_oot)}")

    todos_resultados = []

    for nombre, factory in MODELOS.items():
        print(f"\n{'─'*65}")
        print(f"  METODO: {nombre}")
        print(f"{'─'*65}")

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        modelo_nn = factory()

        if nombre == "1_identidad":
            val_loss, test_loss, ep = 0.0, 0.0, 0
        else:
            val_loss, test_loss, ep = entrenar_nn(modelo_nn, X, Y)

        print(f"  NN — val_loss: {val_loss:.6f} | test_loss: {test_loss:.6f} | epochs: {ep}")

        modelo_nn.eval()
        resultados = evaluar_con_proyeccion(modelo_nn, tucker_model, d, tripletas_oot, emb_oot)
        met = calcular_metricas(resultados)

        met["metodo"] = nombre
        met["nn_val_loss"] = val_loss
        met["nn_test_loss"] = test_loss
        met["nn_epochs"] = ep
        todos_resultados.append(met)

        print(f"  Hits@1={met['hits@1']:.4f} | MRR={met['mrr']:.4f} | Acc={met['acc']:.4f}")
        print(f"  Reprueba -> Prec={met['prec_reprueba']:.4f} | Rec={met['recall_reprueba']:.4f} | F1={met['f1_reprueba']:.4f}")
        print(f"  TP={met['TP']} FP={met['FP']} FN={met['FN']} TN={met['TN']}")

    # Resumen
    print(f"\n{'='*65}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*65}")
    df = pd.DataFrame(todos_resultados)
    cols = ["metodo", "hits@1", "mrr", "acc", "prec_reprueba", "recall_reprueba", "f1_reprueba",
            "TP", "FP", "FN", "TN", "nn_val_loss"]
    print(df[cols].to_string(index=False))

    # Guardar
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "comparacion_proyecciones.csv", index=False, encoding="utf-8-sig")
    print(f"\nGuardado en {OUTPUT_DIR / 'comparacion_proyecciones.csv'}")


if __name__ == "__main__":
    main()
