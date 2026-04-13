"""
Verifica que el orden del vocab de embeddings coincide exactamente con
el orden que construye Data al cargar el dataset S3 (con reverse=True).
"""
import sys, json
sys.path.insert(0, r"C:\Users\LuisPlaza\MEMORIA\tucker_pull")
import torch
from load_data import Data

EMB_PATH = r"C:\Users\LuisPlaza\MEMORIA\tucker_pull\notebooks\Experimento_warm_start\embeddings_8d_binario_s3\embeddings_inicializados_binarios_8d.pt"
VOC_PATH = r"C:\Users\LuisPlaza\MEMORIA\tucker_pull\notebooks\Experimento_warm_start\embeddings_8d_binario_s3\vocabulario_binario_8d.json"
DATA_DIR = r"C:\Users\LuisPlaza\MEMORIA\tucker_pull\data\dataset_2019_2020_2021_hacia_s3_binario/"

# --- Vocab del script de embeddings ---
with open(VOC_PATH, encoding="utf-8") as f:
    vocab = json.load(f)
emb_entities = vocab["entities"]
emb_ent_idxs = {e: i for i, e in enumerate(emb_entities)}
emb_tensor   = torch.load(EMB_PATH, map_location="cpu")
print(f"Vocab embeddings : {len(emb_entities)} entidades | tensor shape: {tuple(emb_tensor.shape)}")

# --- Data como lo instancia el script de entrenamiento ---
d = Data(data_dir=DATA_DIR, reverse=True)
data_ent_idxs = {e: i for i, e in enumerate(d.entities)}
print(f"Data (reverse=T) : {len(d.entities)} entidades | {len(d.relations)} relaciones")

# 1) Conjuntos
s_emb  = set(emb_entities)
s_data = set(d.entities)
solo_emb  = s_emb  - s_data
solo_data = s_data - s_emb
print(f"\n[1] Solo en vocab-emb (no en Data) : {len(solo_emb)}  -> {list(solo_emb)[:5]}")
print(f"    Solo en Data (no en vocab-emb)  : {len(solo_data)} -> {list(solo_data)[:5]}")

# 2) Orden idéntico posición a posición
n = min(len(emb_entities), len(d.entities))
mismatches = [(i, emb_entities[i], d.entities[i])
              for i in range(n) if emb_entities[i] != d.entities[i]]
order_ok = (len(mismatches) == 0 and len(emb_entities) == len(d.entities))
print(f"\n[2] Longitudes iguales : {len(emb_entities) == len(d.entities)} "
      f"({len(emb_entities)} vs {len(d.entities)})")
print(f"    Orden idéntico      : {order_ok}")
if mismatches:
    print(f"    Primeros 5 desajustes (idx, emb, data): {mismatches[:5]}")

# 3) Shape compatible con EDIM=8
print(f"\n[3] Forma tensor : {tuple(emb_tensor.shape)}  "
      f"(esperado [{len(emb_entities)}, 8])")
shape_ok = (emb_tensor.shape == (len(emb_entities), 8))
print(f"    Shape correcto : {shape_ok}")

# 4) Muestra: 5 alumnos — verificar que idx_emb == idx_data y que el vector tenga sentido
print("\n[4] Muestra de 5 alumnos:")
print(f"    {'ID':>12}  idx_emb  idx_data  match  vec (normalizado, primeras 4 dims S1)")
sample = [e for e in d.entities if e.isdigit()][:5]
for ent in sample:
    idx_data = data_ent_idxs.get(ent, -1)
    idx_emb  = emb_ent_idxs.get(ent, -1)
    if idx_emb >= 0:
        vec = [round(v, 3) for v in emb_tensor[idx_emb].tolist()]
    else:
        vec = None
    match = (idx_emb == idx_data)
    print(f"    {ent:>12}  {idx_emb:>7}  {idx_data:>8}  {str(match):>5}  {vec[:4] if vec else 'N/A'}")

# 5) Muestra: 3 relaciones para confirmar
print(f"\n[5] Relaciones en Data  : {d.relations}")
print(f"    Relaciones en vocab : {vocab['relations']}")

# --- Conclusión ---
print("\n" + "="*55)
if order_ok and shape_ok and not solo_emb and not solo_data:
    print("✅  WARM START CORRECTO: vocab y orden coinciden exactamente.")
else:
    print("❌  PROBLEMA DETECTADO — revisar los puntos marcados arriba.")
