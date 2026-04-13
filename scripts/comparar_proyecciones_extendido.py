# -*- coding: utf-8 -*-
"""
Experimento extendido: KNN + variantes adicionales.
Alumnos OOT NO estan en vocab Tucker => scoring manual.
"""
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

# ── Rutas ─────────────────────────────────────────────────────────────
DF_BASE      = REPO_ROOT / "data" / "dataframes_por_semestre"
EMB_INIT     = (REPO_ROOT / "notebooks" / "Experimento_warm_start"
                / "embeddings_8d_binario_s3" / "embeddings_inicializados_binarios_8d.pt")
MODEL_PATH   = REPO_ROOT / "results" / "prueba_s3_8d_rdim11" / "best_model.pt"
VOCAB_PATH   = (REPO_ROOT / "notebooks" / "Experimento_warm_start"
                / "embeddings_8d_binario_s3" / "vocabulario_binario_8d.json")
OUTPUT_DIR   = REPO_ROOT / "results" / "comparacion_proyecciones_8d"

SEM_S1_OOT, SEM_S2_OOT, SEM_OBJ_OOT = "20221", "20222", "20231"
CURSOS_S1 = ["MA1101", "MA1001", "FI1000", "BT1211"]
CURSOS_S2 = ["MA1002", "MA1102", "FI1100", "CC1002"]
CURSOS_S3 = ["MA2001", "MA2601", "FI2001", "FI2003", "IQ2211"]
CURSOS_8D = CURSOS_S1 + CURSOS_S2
CURSOS_SET = set(CURSOS_S1 + CURSOS_S2 + CURSOS_S3)
FAIL_GRADE, APPROVED_TRANSFER = 1.0, 4.0
EDIM, RDIM, SEED = 8, 11, 42
torch.manual_seed(SEED); np.random.seed(SEED)


# ═════════════════ FUNCIONES DATOS ═════════════════════════════════
def _norm_cols(df):
    df = df.copy()
    df["ID"]           = df["ID"].astype(str).str.strip().str.upper()
    df["CURSO"]        = df["CURSO"].astype(str).str.strip().str.upper()
    df["ESTADO_CURSO"] = df["ESTADO_CURSO"].astype(str)
    return df

def _nota_efectiva(estado, nota_raw):
    if "Aprobado (T)" in str(estado) or "Aprobado(T)" in str(estado):
        return APPROVED_TRANSFER
    try: return float(str(nota_raw).replace(",", "."))
    except Exception:
        return APPROVED_TRANSFER if "Aprobado" in str(estado) else FAIL_GRADE

def _normalizar(n):
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return (FAIL_GRADE - 4.0) / 3.0
    return (float(n) - 4.0) / 3.0

def _construir_pivot(sem, cursos):
    ruta = DF_BASE / f"df_{sem}.csv"
    if not ruta.exists(): return pd.DataFrame(columns=cursos)
    df = _norm_cols(pd.read_csv(ruta, sep=";"))
    df = df[df["CURSO"].isin(cursos)].copy()
    df["NOTA_EF"] = [_nota_efectiva(e, n) for e, n in zip(df["ESTADO_CURSO"], df["NOTA"])]
    pivot = df.pivot_table(index="ID", columns="CURSO", values="NOTA_EF", aggfunc="max")
    return pivot.reindex(columns=cursos)

def _determinar_relacion(estado, nota_raw):
    e = str(estado)
    if "Aprobado" in e and "Eliminado" not in e and "Reprobado" not in e: return "aprueba"
    if "Reprobado" in e or "Eliminado" in e: return "reprueba"
    try:
        nota = float(str(nota_raw).replace(",", "."))
        return "reprueba" if nota < 4.0 else "aprueba"
    except: return None

def construir_emb_8d(ids, sem_s1, sem_s2):
    p1 = _construir_pivot(sem_s1, CURSOS_S1).fillna(FAIL_GRADE)
    p2 = _construir_pivot(sem_s2, CURSOS_S2).fillna(FAIL_GRADE)
    p1, p2 = p1[p1.index.isin(ids)], p2[p2.index.isin(ids)]
    tabla = p1.join(p2, how="outer").fillna(FAIL_GRADE).reindex(columns=CURSOS_8D)
    result = {}
    for a in ids:
        if a in tabla.index:
            row = tabla.loc[a]
            result[a] = np.array([_normalizar(row[c]) for c in CURSOS_8D], dtype=np.float32)
        else:
            result[a] = np.array([_normalizar(FAIL_GRADE)] * 8, dtype=np.float32)
    return result


# ═════════════════ CARGAR TUCKER ═══════════════════════════════════
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


# ═════════════════ MODELOS NN ══════════════════════════════════════
class ResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 8),
        )
    def forward(self, x):
        return x + self.net(x)

class MLPOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 8),
        )
    def forward(self, x):
        return self.net(x)

class ResidualNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 8),
        )
        self.ln = nn.LayerNorm(8)
    def forward(self, x):
        return self.ln(x + self.net(x))


def entrenar_nn(modelo_nn, X, Y, epochs=500, lr=1e-3, patience=200, loss_type="mse"):
    n = len(X)
    idx = np.random.permutation(n)
    n_tr, n_va = int(0.8 * n), int(0.15 * n)
    X_tr, Y_tr = X[idx[:n_tr]], Y[idx[:n_tr]]
    X_va, Y_va = X[idx[n_tr:n_tr+n_va]], Y[idx[n_tr:n_tr+n_va]]
    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=64, shuffle=True)
    opt = torch.optim.Adam(modelo_nn.parameters(), lr=lr)
    if loss_type == "cosine":
        loss_fn = lambda pred, tgt: 1.0 - nn.functional.cosine_similarity(pred, tgt, dim=1).mean()
    else:
        loss_fn = nn.MSELoss()
    best_val, best_state, no_improve = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        modelo_nn.train()
        for xb, yb in loader:
            opt.zero_grad(); loss_fn(modelo_nn(xb), yb).backward(); opt.step()
        modelo_nn.eval()
        with torch.no_grad():
            vl = float(loss_fn(modelo_nn(X_va), Y_va))
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in modelo_nn.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience: break
    if best_state: modelo_nn.load_state_dict(best_state)
    modelo_nn.eval()
    return best_val, ep


# ═════════════════ KNN ═════════════════════════════════════════════
class KNNProjector:
    def __init__(self, X_train, Y_train, k=10, metric="l2"):
        self.X = X_train.numpy()
        self.Y = Y_train.numpy()
        self.k = k
        self.metric = metric

    def project(self, x_new):
        if self.metric == "cosine":
            norms_x = np.linalg.norm(self.X, axis=1, keepdims=True)
            norms_x = np.maximum(norms_x, 1e-8)
            norm_new = max(np.linalg.norm(x_new), 1e-8)
            sim = (self.X / norms_x) @ (x_new / norm_new)
            nearest = np.argsort(-sim)[:self.k]
            w = np.maximum(sim[nearest], 0.0) + 1e-8
        else:
            dists = np.linalg.norm(self.X - x_new, axis=1)
            nearest = np.argsort(dists)[:self.k]
            d = np.maximum(dists[nearest], 1e-8)
            w = 1.0 / d
        w = w / (w.sum() + 1e-12)
        return (w[:, None] * self.Y[nearest]).sum(axis=0).astype(np.float32)


# ═════════════════ SCORING MANUAL ══════════════════════════════════
def evaluar_manual(tucker_model, d, tripletas, emb_dict):
    rels_fwd = [r for r in d.relations if not r.endswith("_reverse")]
    rel_idxs_fwd = {r: d.relation_idxs[r] for r in rels_fwd}
    W_flat = tucker_model.W.data.view(RDIM, EDIM * EDIM)
    W_mats = {}
    with torch.no_grad():
        for r_name in rels_fwd:
            e_r = tucker_model.R.weight[rel_idxs_fwd[r_name]]
            W_mats[r_name] = (e_r @ W_flat).view(EDIM, EDIM)

    resultados = []
    with torch.no_grad():
        for alumno, rel_true, curso in tripletas:
            if curso not in d.entity_idxs or alumno not in emb_dict:
                continue
            e_h = torch.tensor(emb_dict[alumno], dtype=torch.float32).unsqueeze(0)
            e_h_bn = tucker_model.bn0(e_h).squeeze(0)
            t_idx = d.entity_idxs[curso]
            e_t = tucker_model.E.weight[t_idx]
            scores = {}
            for r_name in rels_fwd:
                xr = e_h_bn @ W_mats[r_name]
                xr = tucker_model.bn1(xr.unsqueeze(0)).squeeze(0)
                s = float(torch.sigmoid(xr @ e_t))
                scores[r_name] = s
            orden = sorted(rels_fwd, key=lambda r: scores[r], reverse=True)
            pred_rel = orden[0]
            rank = orden.index(rel_true) + 1 if rel_true in orden else len(rels_fwd) + 1
            resultados.append({
                "alumno": alumno, "rel_true": rel_true, "rel_pred": pred_rel,
                "curso": curso, "rank": rank,
            })
    return resultados


def calcular_metricas(resultados):
    if not resultados:
        return {"hits@1":0,"mrr":0,"acc":0,"prec_reprueba":0,
                "recall_reprueba":0,"f1_reprueba":0,"TP":0,"FP":0,"FN":0,"TN":0,"n":0}
    ranks = np.array([r["rank"] for r in resultados], dtype=float)
    rel_true = [r["rel_true"] for r in resultados]
    rel_pred = [r["rel_pred"] for r in resultados]
    n = len(ranks)
    hits1 = float(np.mean(ranks == 1))
    mrr   = float(np.mean(1.0 / ranks))
    TP = sum(1 for t,p in zip(rel_true, rel_pred) if t=="reprueba" and p=="reprueba")
    FP = sum(1 for t,p in zip(rel_true, rel_pred) if t=="aprueba" and p=="reprueba")
    FN = sum(1 for t,p in zip(rel_true, rel_pred) if t=="reprueba" and p=="aprueba")
    TN = sum(1 for t,p in zip(rel_true, rel_pred) if t=="aprueba" and p=="aprueba")
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    acc  = (TP+TN)/n if n>0 else 0.0
    return {"hits@1":hits1,"mrr":mrr,"acc":acc,
            "prec_reprueba":prec,"recall_reprueba":rec,"f1_reprueba":f1,
            "TP":TP,"FP":FP,"FN":FN,"TN":TN,"n":n}


# ═════════════════ MAIN ════════════════════════════════════════════
def main():
    print("=" * 65)
    print("EXPERIMENTO EXTENDIDO: KNN + COSINE + RESIDUAL-NORM")
    print("=" * 65)

    tucker_model, d = cargar_tucker()
    init_tensor = torch.load(EMB_INIT, map_location="cpu")
    learned     = tucker_model.E.weight.data
    alum_idxs = [i for i, e in enumerate(d.entities) if e not in CURSOS_SET]
    X_all, Y_all = init_tensor[alum_idxs], learned[alum_idxs]
    print(f"Alumnos train: {len(alum_idxs)}")

    diff = Y_all - X_all
    print(f"Drift L2 promedio: {float(diff.norm(dim=1).mean()):.4f}")
    print(f"Ratio drift/init:  {float(diff.norm(dim=1).mean() / X_all.norm(dim=1).mean()):.4f}")

    # Tripletas OOT
    df_s1 = _norm_cols(pd.read_csv(DF_BASE / f"df_{SEM_S1_OOT}.csv", sep=";"))
    al4 = df_s1[df_s1["CURSO"].isin(CURSOS_S1)].groupby("ID")["CURSO"].nunique()
    ids_v = set(al4[al4 == 4].index)
    df_obj = _norm_cols(pd.read_csv(DF_BASE / f"df_{SEM_OBJ_OOT}.csv", sep=";"))
    df_obj = df_obj[df_obj["ID"].isin(ids_v) & df_obj["CURSO"].isin(CURSOS_SET)].copy()
    df_obj["relacion"] = [_determinar_relacion(e, n)
                          for e, n in zip(df_obj["ESTADO_CURSO"], df_obj["NOTA"])]
    df_obj = df_obj.dropna(subset=["relacion"])
    tripletas = list(zip(df_obj["ID"], df_obj["relacion"], df_obj["CURSO"]))
    ids_oot = list(df_obj["ID"].unique())
    emb_oot = construir_emb_8d(ids_oot, SEM_S1_OOT, SEM_S2_OOT)
    n_apr = sum(1 for _,r,_ in tripletas if r=="aprueba")
    n_rep = sum(1 for _,r,_ in tripletas if r=="reprueba")
    print(f"Tripletas OOT: {len(tripletas)} | Alumnos: {len(ids_oot)}")
    print(f"  aprueba: {n_apr} ({n_apr/len(tripletas)*100:.1f}%) | reprueba: {n_rep} ({n_rep/len(tripletas)*100:.1f}%)")

    todos = []

    metodos = [
        ("1_identidad",       "identity", None),
        ("2_mlp_MSE",         "nn", ("mlp","mse")),
        ("3_residual_MSE",    "nn", ("residual","mse")),
        ("4_mlp_cosine",      "nn", ("mlp","cosine")),
        ("5_residual_norm",   "nn", ("resnorm","mse")),
        ("6_knn_k5_l2",       "knn", (5,"l2")),
        ("7_knn_k10_l2",      "knn", (10,"l2")),
        ("8_knn_k20_l2",      "knn", (20,"l2")),
        ("9_knn_k10_cosine",  "knn", (10,"cosine")),
        ("10_knn_k20_cosine", "knn", (20,"cosine")),
    ]

    for nombre, tipo, params in metodos:
        print(f"\n{'─'*65}")
        print(f"  {nombre}")
        print(f"{'─'*65}")

        if tipo == "identity":
            emb_proj = {a: emb_oot[a] for a in ids_oot}
        elif tipo == "nn":
            arch, loss_t = params
            torch.manual_seed(SEED); np.random.seed(SEED)
            if arch == "mlp":     nn_m = MLPOriginal()
            elif arch == "residual": nn_m = ResidualModel()
            elif arch == "resnorm":  nn_m = ResidualNorm()
            val_loss, ep = entrenar_nn(nn_m, X_all, Y_all, loss_type=loss_t)
            print(f"  val_loss={val_loss:.6f}, epochs={ep}")
            nn_m.eval()
            emb_proj = {}
            with torch.no_grad():
                for a in ids_oot:
                    x = torch.tensor(emb_oot[a]).unsqueeze(0)
                    emb_proj[a] = nn_m(x).squeeze(0).numpy()
        elif tipo == "knn":
            k, metric = params
            knn = KNNProjector(X_all, Y_all, k=k, metric=metric)
            emb_proj = {a: knn.project(emb_oot[a]) for a in ids_oot}

        resultados = evaluar_manual(tucker_model, d, tripletas, emb_proj)
        met = calcular_metricas(resultados)
        met["metodo"] = nombre
        todos.append(met)
        print(f"  Hits@1={met['hits@1']:.4f} | MRR={met['mrr']:.4f} | Acc={met['acc']:.4f}")
        print(f"  Prec={met['prec_reprueba']:.4f} | Rec={met['recall_reprueba']:.4f} | F1={met['f1_reprueba']:.4f}")
        print(f"  TP={met['TP']} FP={met['FP']} FN={met['FN']} TN={met['TN']}")

    print(f"\n{'='*65}")
    print("TABLA COMPARATIVA FINAL")
    print(f"{'='*65}")
    df = pd.DataFrame(todos)
    cols = ["metodo","hits@1","mrr","acc","prec_reprueba","recall_reprueba","f1_reprueba","TP","FP","FN","TN"]
    print(df[cols].to_string(index=False))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "comparacion_extendida.csv", index=False, encoding="utf-8-sig")
    print(f"\nGuardado en {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
