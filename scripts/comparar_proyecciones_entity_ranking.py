# -*- coding: utf-8 -*-
"""
Comparación de proyecciones con ENTITY RANKING (filtered setting).
Para cada (alumno_OOT, relación, curso), rankea TODAS las entidades como posible tail.
"""
import sys, json
from pathlib import Path
from collections import defaultdict

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
# Usamos el modelo entrenado con entity ranking
MODEL_PATH   = REPO_ROOT / "results" / "prueba_s3_8d_rdim11_entity_ranking" / "best_model.pt"
VOCAB_PATH   = (REPO_ROOT / "notebooks" / "Experimento_warm_start"
                / "embeddings_8d_binario_s3" / "vocabulario_binario_8d.json")
OUTPUT_DIR   = REPO_ROOT / "results" / "comparacion_proyecciones_8d_entity_ranking"

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


# ═════════════════ ENTITY RANKING (filtered) ═══════════════════════
def evaluar_entity_ranking(tucker_model, d, tripletas, emb_dict):
    """
    Entity ranking con filtered setting para alumnos OOT.
    Para cada (alumno, relacion, curso), rankea TODAS las entidades como tail.
    El forward manual replica model.forward() usando el embedding proyectado.
    """
    tucker_model.eval()
    n_entities = len(d.entities)

    # Filtered setting: construir er_vocab de las tripletas OOT
    # (alumno, relacion) -> [lista de curso_idx validos]
    er_vocab = defaultdict(list)
    for alumno, rel, curso in tripletas:
        if curso in d.entity_idxs:
            er_vocab[(alumno, rel)].append(d.entity_idxs[curso])

    hits = [[] for _ in range(10)]
    ranks = []
    resultados = []

    with torch.no_grad():
        # Pre-computar W_r para cada relación (sin _reverse)
        rels_fwd = [r for r in d.relations if not r.endswith("_reverse")]
        W_flat = tucker_model.W.data.view(RDIM, EDIM * EDIM)

        for alumno, rel_true, curso in tripletas:
            if alumno not in emb_dict or curso not in d.entity_idxs:
                continue
            if rel_true not in d.relation_idxs:
                continue

            t_idx = d.entity_idxs[curso]
            r_idx = d.relation_idxs[rel_true]

            # Forward manual: replica model.forward(e1_idx, r_idx)
            e_h = torch.tensor(emb_dict[alumno], dtype=torch.float32).unsqueeze(0)  # 1 x d_e
            x = tucker_model.bn0(e_h)     # 1 x d_e (dropout no-op en eval)
            x = x.view(1, 1, EDIM)        # 1 x 1 x d_e

            r = tucker_model.R.weight[r_idx].unsqueeze(0)  # 1 x d_r
            W_mat = torch.mm(r, tucker_model.W.view(RDIM, -1))  # 1 x (d_e*d_e)
            W_mat = W_mat.view(1, EDIM, EDIM)  # 1 x d_e x d_e

            x = torch.bmm(x, W_mat)      # 1 x 1 x d_e
            x = x.view(1, EDIM)           # 1 x d_e
            x = tucker_model.bn1(x)       # 1 x d_e
            predictions = torch.sigmoid(torch.mm(x, tucker_model.E.weight.T))  # 1 x n_entities
            predictions = predictions.squeeze(0)  # n_entities

            # Filtered setting: zero out other valid tails
            filt = er_vocab.get((alumno, rel_true), [])
            target_value = predictions[t_idx].item()
            if filt:
                predictions[filt] = 0.0
            predictions[t_idx] = target_value

            # Rank
            sort_idxs = torch.argsort(predictions, descending=True).numpy()
            rank = int(np.where(sort_idxs == t_idx)[0][0]) + 1

            ranks.append(rank)
            for k in range(10):
                hits[k].append(1.0 if rank <= (k + 1) else 0.0)

            resultados.append({
                "alumno": alumno, "rel_true": rel_true, "curso": curso,
                "rank": rank, "score": target_value,
                "pred_entity": d.entities[sort_idxs[0]],
            })

    return resultados, hits, ranks


def calcular_metricas_entity(resultados, hits, ranks):
    if not resultados:
        return {"hits@1": 0, "hits@3": 0, "hits@10": 0, "mr": 0, "mrr": 0, "n": 0}
    ranks = np.array(ranks, dtype=float)
    return {
        "hits@1":  float(np.mean(hits[0])),
        "hits@3":  float(np.mean(hits[2])),
        "hits@10": float(np.mean(hits[9])),
        "mr":      float(np.mean(ranks)),
        "mrr":     float(np.mean(1.0 / ranks)),
        "n":       len(ranks),
    }


# ═════════════════ MAIN ════════════════════════════════════════════
def main():
    print("=" * 65)
    print("ENTITY RANKING: COMPARACION DE PROYECCIONES (FILTERED SETTING)")
    print("=" * 65)

    tucker_model, d = cargar_tucker()
    init_tensor = torch.load(EMB_INIT, map_location="cpu")
    learned     = tucker_model.E.weight.data
    alum_idxs = [i for i, e in enumerate(d.entities) if e not in CURSOS_SET]
    X_all, Y_all = init_tensor[alum_idxs], learned[alum_idxs]
    print(f"Alumnos train (para NN/KNN): {len(alum_idxs)}")

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
    n_apr = sum(1 for _, r, _ in tripletas if r == "aprueba")
    n_rep = sum(1 for _, r, _ in tripletas if r == "reprueba")
    print(f"Tripletas OOT: {len(tripletas)} | Alumnos: {len(ids_oot)}")
    print(f"  aprueba: {n_apr} ({n_apr/len(tripletas)*100:.1f}%) | "
          f"reprueba: {n_rep} ({n_rep/len(tripletas)*100:.1f}%)")
    print(f"Entidades en vocabulario: {len(d.entities)}")

    todos = []

    metodos = [
        ("1_identidad",       "identity", None),
        ("2_mlp_MSE",         "nn", ("mlp", "mse")),
        ("3_residual_MSE",    "nn", ("residual", "mse")),
        ("4_mlp_cosine",      "nn", ("mlp", "cosine")),
        ("5_residual_norm",   "nn", ("resnorm", "mse")),
        ("6_knn_k5_l2",       "knn", (5, "l2")),
        ("7_knn_k10_l2",      "knn", (10, "l2")),
        ("8_knn_k20_l2",      "knn", (20, "l2")),
        ("9_knn_k10_cosine",  "knn", (10, "cosine")),
        ("10_knn_k20_cosine", "knn", (20, "cosine")),
    ]

    for nombre, tipo, params in metodos:
        print(f"\n{'─' * 65}")
        print(f"  {nombre}")
        print(f"{'─' * 65}")

        if tipo == "identity":
            emb_proj = {a: emb_oot[a] for a in ids_oot}
        elif tipo == "nn":
            arch, loss_t = params
            torch.manual_seed(SEED); np.random.seed(SEED)
            if arch == "mlp":        nn_m = MLPOriginal()
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

        resultados, hits, ranks = evaluar_entity_ranking(tucker_model, d, tripletas, emb_proj)
        met = calcular_metricas_entity(resultados, hits, ranks)
        met["metodo"] = nombre
        todos.append(met)
        print(f"  Hits@1={met['hits@1']:.4f} | Hits@3={met['hits@3']:.4f} | "
              f"Hits@10={met['hits@10']:.4f}")
        print(f"  MR={met['mr']:.2f} | MRR={met['mrr']:.4f} | n={met['n']}")

    print(f"\n{'=' * 65}")
    print("TABLA COMPARATIVA FINAL (ENTITY RANKING)")
    print(f"{'=' * 65}")
    df = pd.DataFrame(todos)
    cols = ["metodo", "hits@1", "hits@3", "hits@10", "mr", "mrr", "n"]
    print(df[cols].to_string(index=False))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "comparacion_entity_ranking.csv", index=False, encoding="utf-8-sig")
    print(f"\nGuardado en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
