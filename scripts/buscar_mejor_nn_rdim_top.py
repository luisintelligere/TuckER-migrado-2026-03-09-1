"""
Busqueda de arquitectura optima de la MLP proyectora para rdim=5 y rdim=8
(los dos mejores segun metricas OOT del experimento augmentado).

Varia: hidden layers, dropout, learning rate, activacion.
Evalua en cohortes OOT 20221->20222 y 20231->20232.
Reporta mejor config por F1 y por Recall (relacion reprueba).
"""
from __future__ import annotations

import sys
from pathlib import Path
from itertools import product

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import TuckER
from load_data import Data

DF_BASE = REPO_ROOT / "data" / "dataframes_por_semestre"
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "busqueda_nn_rdim_top.csv"

CURSOS_S1 = ["BT1211", "FI1000", "MA1001", "MA1101"]
CURSOS_S2 = ["CC1002", "FI1100", "MA1002", "MA1102"]
CURSOS_PERMITIDOS = set(CURSOS_S1 + CURSOS_S2)

EDIM = 4
FAIL_GRADE = 1.0
APPROVED_TRANSFER = 4.0
SEED = 42

DATASET_NAME = "dataset_augmentado_1920reprueba_2021_s2"
TRAIN_COHORTES = [("20191", "20192"), ("20201", "20202"), ("20211", "20212")]
OOT_COHORTES = [("20221", "20222"), ("20231", "20232")]

# rdims de interes (mejores metricas OOT)
TARGET_RDIMS = [5, 8]

# -----------------------------------------------------------------------
# Espacio de busqueda
# -----------------------------------------------------------------------
CONFIGS = [
    # --- baseline ---
    {"name": "baseline",       "hidden": [64, 128],       "dropout": 0.3, "lr": 1e-3, "act": "relu"},
    # --- mas anchas ---
    {"name": "wider",          "hidden": [128, 256],      "dropout": 0.3, "lr": 1e-3, "act": "relu"},
    {"name": "large",          "hidden": [256, 512],      "dropout": 0.4, "lr": 1e-3, "act": "relu"},
    # --- mas pequenas ---
    {"name": "small",          "hidden": [32, 64],        "dropout": 0.2, "lr": 1e-3, "act": "relu"},
    {"name": "tiny",           "hidden": [16, 32],        "dropout": 0.1, "lr": 1e-3, "act": "relu"},
    # --- capa simple ---
    {"name": "single_128",     "hidden": [128],           "dropout": 0.2, "lr": 1e-3, "act": "relu"},
    {"name": "single_256",     "hidden": [256],           "dropout": 0.3, "lr": 1e-3, "act": "relu"},
    # --- mas profundas ---
    {"name": "deep_sym",       "hidden": [64, 128, 64],   "dropout": 0.3, "lr": 1e-3, "act": "relu"},
    {"name": "deep_wide",      "hidden": [128, 256, 128], "dropout": 0.3, "lr": 1e-3, "act": "relu"},
    {"name": "deep_small",     "hidden": [32, 64, 32],    "dropout": 0.2, "lr": 1e-3, "act": "relu"},
    # --- variando dropout (baseline arch) ---
    {"name": "drop0",          "hidden": [64, 128],       "dropout": 0.0, "lr": 1e-3, "act": "relu"},
    {"name": "drop01",         "hidden": [64, 128],       "dropout": 0.1, "lr": 1e-3, "act": "relu"},
    {"name": "drop05",         "hidden": [64, 128],       "dropout": 0.5, "lr": 1e-3, "act": "relu"},
    # --- variando lr (baseline arch) ---
    {"name": "lr5e4",          "hidden": [64, 128],       "dropout": 0.3, "lr": 5e-4, "act": "relu"},
    {"name": "lr1e4",          "hidden": [64, 128],       "dropout": 0.3, "lr": 1e-4, "act": "relu"},
    # --- activacion tanh ---
    {"name": "tanh_base",      "hidden": [64, 128],       "dropout": 0.3, "lr": 1e-3, "act": "tanh"},
    {"name": "tanh_wider",     "hidden": [128, 256],      "dropout": 0.2, "lr": 1e-3, "act": "tanh"},
    {"name": "tanh_deep",      "hidden": [64, 128, 64],   "dropout": 0.2, "lr": 5e-4, "act": "tanh"},
    # --- activacion leaky relu ---
    {"name": "leaky_base",     "hidden": [64, 128],       "dropout": 0.3, "lr": 1e-3, "act": "leaky"},
    {"name": "leaky_wider",    "hidden": [128, 256],      "dropout": 0.2, "lr": 1e-3, "act": "leaky"},
    # --- combos prometedores ---
    {"name": "combo_nodrop",   "hidden": [128, 256],      "dropout": 0.1, "lr": 5e-4, "act": "relu"},
    {"name": "combo_tanh_lr",  "hidden": [64, 128, 64],   "dropout": 0.1, "lr": 5e-4, "act": "tanh"},
    {"name": "combo_wide_lr",  "hidden": [256, 512],      "dropout": 0.2, "lr": 5e-4, "act": "relu"},
]


# -----------------------------------------------------------------------
# Helpers reutilizados del script base
# -----------------------------------------------------------------------

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


def cargar_tucker(rdim):
    folder = RESULTS_DIR / f"s2_ws4d_{DATASET_NAME}_rdim{rdim}"
    model_path = folder / "best_model.pt"
    data_dir = f"data/{DATASET_NAME}/"
    d = Data(data_dir=data_dir, reverse=True)
    model = TuckER(d, EDIM, rdim, input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    entity_idxs = {e: i for i, e in enumerate(d.entities)}
    relation_idxs = {r: i for i, r in enumerate(d.relations)}
    return model, d, entity_idxs, relation_idxs


def preparar_datos_nn(model, d, entity_idxs):
    learned = model.E.weight.data
    alum_entities = [
        e for e in d.entities
        if e not in CURSOS_PERMITIDOS
        and not any(e == c + "_REVERSE" for c in CURSOS_PERMITIDOS)
    ]
    notas_por_alumno = {}
    for sem_s1, _ in TRAIN_COHORTES:
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
    return X, Y


def generar_tripletas_oot(sem_s1, sem_s2):
    ruta_s1 = DF_BASE / f"df_{sem_s1}.csv"
    ruta_s2 = DF_BASE / f"df_{sem_s2}.csv"
    if not ruta_s1.exists() or not ruta_s2.exists():
        return [], []
    df_s1 = _norm_cols(pd.read_csv(ruta_s1, sep=";"))
    df_s2 = _norm_cols(pd.read_csv(ruta_s2, sep=";"))
    conteo = df_s1[df_s1["CURSO"].isin(CURSOS_S1)].groupby("ID")["CURSO"].nunique()
    ids_con_4 = set(conteo[conteo == 4].index)
    df_s2_filt = df_s2[
        df_s2["ID"].isin(ids_con_4) & df_s2["CURSO"].isin(CURSOS_PERMITIDOS)
    ].copy()
    df_s2_filt["relacion"] = [
        _determinar_relacion(e, n)
        for e, n in zip(df_s2_filt["ESTADO_CURSO"], df_s2_filt["NOTA"])
    ]
    df_s2_filt = df_s2_filt.dropna(subset=["relacion"])
    tripletas = list(zip(df_s2_filt["ID"], df_s2_filt["relacion"], df_s2_filt["CURSO"]))
    ids_unicos = df_s2_filt["ID"].unique().tolist()
    return tripletas, ids_unicos


# -----------------------------------------------------------------------
# MLP flexible
# -----------------------------------------------------------------------

class MLPFlexible(nn.Module):
    """MLP configurable: hidden layers, dropout, activacion."""

    def __init__(self, input_dim, output_dim, hidden, dropout, act):
        super().__init__()
        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky": lambda: nn.LeakyReLU(0.1),
        }
        act_fn = act_map.get(act, nn.ReLU)

        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), act_fn()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def entrenar_nn(X, Y, cfg, edim, epochs=500, patience=100):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    modelo = MLPFlexible(4, edim, cfg["hidden"], cfg["dropout"], cfg["act"])
    n = len(X)
    idx = np.random.permutation(n)
    n_tr = int(0.8 * n)
    n_va = int(0.15 * n)
    X_tr, Y_tr = X[idx[:n_tr]], Y[idx[:n_tr]]
    X_va, Y_va = X[idx[n_tr:n_tr + n_va]], Y[idx[n_tr:n_tr + n_va]]

    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=64, shuffle=True)
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0
    ep = 0

    for ep in range(1, epochs + 1):
        modelo.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(modelo(xb), yb).backward()
            opt.step()
        modelo.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(modelo(X_va), Y_va))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in modelo.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state:
        modelo.load_state_dict(best_state)
    modelo.eval()
    return modelo, best_val, ep


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
            scores = {r: float((e_h_bn @ W_mats[r]) @ e_t) for r in rels_fwd}
            orden = sorted(rels_fwd, key=lambda r: scores[r], reverse=True)
            rank = orden.index(rel_true) + 1 if rel_true in orden else len(rels_fwd) + 1
            resultados.append({
                "rel_true": rel_true,
                "rel_pred": orden[0],
                "rank": rank,
            })
    return resultados


def calcular_metricas(resultados):
    if not resultados:
        return {}
    ranks = np.array([r["rank"] for r in resultados], dtype=float)
    rel_true = [r["rel_true"] for r in resultados]
    rel_pred = [r["rel_pred"] for r in resultados]
    n = len(ranks)
    TP = sum(1 for t, p in zip(rel_true, rel_pred) if t == "reprueba" and p == "reprueba")
    FP = sum(1 for t, p in zip(rel_true, rel_pred) if t == "aprueba"  and p == "reprueba")
    FN = sum(1 for t, p in zip(rel_true, rel_pred) if t == "reprueba" and p == "aprueba")
    TN = sum(1 for t, p in zip(rel_true, rel_pred) if t == "aprueba"  and p == "aprueba")
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "n": n,
        "acc": (TP + TN) / n,
        "prec": prec, "recall": rec, "f1": f1,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "hits1": float(np.mean(ranks == 1)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for rdim in TARGET_RDIMS:
        print(f"\n{'='*70}")
        print(f"  Cargando TuckER  rdim={rdim}")
        print(f"{'='*70}")
        model, d, eidx, ridx = cargar_tucker(rdim)

        # Preparar datos NN una sola vez por rdim
        X, Y = preparar_datos_nn(model, d, eidx)
        print(f"  Datos NN: {len(X)} alumnos")

        # Precargar datos OOT
        oot_data = {}
        for sem_s1, sem_s2 in OOT_COHORTES:
            trips, ids_oot = generar_tripletas_oot(sem_s1, sem_s2)
            emb_notas = construir_notas_s1(ids_oot, sem_s1)
            oot_data[(sem_s1, sem_s2)] = (trips, emb_notas)
            print(f"  OOT {sem_s1}->{sem_s2}: {len(trips)} tripletas, {len(ids_oot)} alumnos")

        print(f"\n  Buscando entre {len(CONFIGS)} configuraciones...\n")

        for i, cfg in enumerate(CONFIGS, 1):
            nn_model, val_loss, epochs_run = entrenar_nn(X, Y, cfg, EDIM)

            for sem_s1, sem_s2 in OOT_COHORTES:
                trips, emb_notas = oot_data[(sem_s1, sem_s2)]
                resultados = evaluar_oot(nn_model, model, eidx, ridx, d,
                                         trips, emb_notas, rdim)
                met = calcular_metricas(resultados)

                row = {
                    "rdim": rdim,
                    "config": cfg["name"],
                    "hidden": str(cfg["hidden"]),
                    "dropout": cfg["dropout"],
                    "lr": cfg["lr"],
                    "act": cfg["act"],
                    "cohorte_oot": f"{sem_s1}->{sem_s2}",
                    "nn_val_loss": round(val_loss, 6),
                    "nn_epochs": epochs_run,
                    **met,
                }
                all_rows.append(row)

            # Resumen rapido por linea
            # Toma el promedio f1 sobre las dos cohortes para este config
            filas_cfg = [r for r in all_rows if r["rdim"] == rdim and r["config"] == cfg["name"]]
            f1_mean  = np.mean([r["f1"]     for r in filas_cfg])
            rec_mean = np.mean([r["recall"] for r in filas_cfg])
            print(
                f"  [{i:02d}/{len(CONFIGS)}] rdim={rdim} | {cfg['name']:20s} | "
                f"val_loss={val_loss:.5f} | epochs={epochs_run:4d} | "
                f"F1_avg={f1_mean:.4f} | Rec_avg={rec_mean:.4f}"
            )

        # --- Resumen para este rdim ---
        df_rdim = pd.DataFrame([r for r in all_rows if r["rdim"] == rdim])

        # Mejor por F1 promedio (entre las dos cohortes)
        f1_avg = df_rdim.groupby("config")["f1"].mean().sort_values(ascending=False)
        rec_avg = df_rdim.groupby("config")["recall"].mean().sort_values(ascending=False)

        print(f"\n  {'='*60}")
        print(f"  TOP-5 por F1 promedio (rdim={rdim})")
        print(f"  {'='*60}")
        for cfg_name in f1_avg.head(5).index:
            rows_c = df_rdim[df_rdim["config"] == cfg_name]
            for _, row in rows_c.iterrows():
                print(
                    f"    {cfg_name:22s} | {row['cohorte_oot']} | "
                    f"F1={row['f1']:.4f} | Prec={row['prec']:.4f} | "
                    f"Rec={row['recall']:.4f} | Acc={row['acc']:.4f}"
                )

        print(f"\n  TOP-5 por Recall promedio (rdim={rdim})")
        print(f"  {'='*60}")
        for cfg_name in rec_avg.head(5).index:
            rows_c = df_rdim[df_rdim["config"] == cfg_name]
            for _, row in rows_c.iterrows():
                print(
                    f"    {cfg_name:22s} | {row['cohorte_oot']} | "
                    f"F1={row['f1']:.4f} | Prec={row['prec']:.4f} | "
                    f"Rec={row['recall']:.4f} | Acc={row['acc']:.4f}"
                )

    # Guardar CSV completo
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResultados completos guardados en: {OUTPUT_CSV}")

    # --- Resumen global ---
    print(f"\n{'='*70}")
    print("  RESUMEN GLOBAL — mejor config por rdim")
    print(f"{'='*70}")
    for rdim in TARGET_RDIMS:
        df_r = df_all[df_all["rdim"] == rdim]
        f1_avg = df_r.groupby("config")["f1"].mean()
        best_f1_name = f1_avg.idxmax()
        rec_avg = df_r.groupby("config")["recall"].mean()
        best_rec_name = rec_avg.idxmax()
        print(f"\n  rdim={rdim}")
        print(f"    Mejor F1     : {best_f1_name}  (F1_avg={f1_avg[best_f1_name]:.4f})")
        print(f"    Mejor Recall : {best_rec_name}  (Rec_avg={rec_avg[best_rec_name]:.4f})")
        # Detalle
        for label, cname in [("F1", best_f1_name), ("Rec", best_rec_name)]:
            if cname == best_f1_name and label == "Rec":
                continue  # ya impreso si son iguales
            rows = df_r[df_r["config"] == cname]
            for _, row in rows.iterrows():
                print(
                    f"      {label} | {row['cohorte_oot']} | "
                    f"F1={row['f1']:.4f} Prec={row['prec']:.4f} "
                    f"Rec={row['recall']:.4f} Acc={row['acc']:.4f} | "
                    f"hidden={row['hidden']} drop={row['dropout']} "
                    f"lr={row['lr']} act={row['act']}"
                )


if __name__ == "__main__":
    main()
