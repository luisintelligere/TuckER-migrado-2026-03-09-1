"""Analiza drift entre embeddings iniciales y aprendidos."""
import torch, json, numpy as np

init = torch.load(r'C:\Users\LuisPlaza\MEMORIA\tucker_pull\notebooks\Experimento_warm_start\embeddings_8d_binario_s3\embeddings_inicializados_binarios_8d.pt', map_location='cpu')
state = torch.load(r'C:\Users\LuisPlaza\MEMORIA\tucker_pull\results\prueba_s3_8d_rdim11\best_model.pt', map_location='cpu')
learned = state['E.weight']

with open(r'C:\Users\LuisPlaza\MEMORIA\tucker_pull\notebooks\Experimento_warm_start\embeddings_8d_binario_s3\vocabulario_binario_8d.json') as f:
    vocab = json.load(f)
ents = vocab['entities']

alum_idxs = [i for i, e in enumerate(ents) if e.startswith('A')]
cursos_idx = [i for i, e in enumerate(ents) if not e.startswith('A')]

diff = (learned - init)[alum_idxs]
print(f'Alumnos: {len(alum_idxs)} | Cursos: {len(cursos_idx)}')
print(f'Norm L2 diff promedio: {float(diff.norm(dim=1).mean()):.4f}')
print(f'Norm L2 init promedio: {float(init[alum_idxs].norm(dim=1).mean()):.4f}')
print(f'Norm L2 learned prom:  {float(learned[alum_idxs].norm(dim=1).mean()):.4f}')
print(f'Max diff absoluto:     {float(diff.abs().max()):.4f}')
print(f'Ratio drift/init:      {float(diff.norm(dim=1).mean()/init[alum_idxs].norm(dim=1).mean()):.4f}')
print()
for d in range(8):
    x = init[alum_idxs, d].numpy()
    y = learned[alum_idxs, d].numpy()
    r = float(np.corrcoef(x, y)[0, 1])
    print(f'  Dim {d}: pearson r = {r:.4f}')
print()
for i in alum_idxs[:3]:
    pad = ' ' * len(ents[i])
    print(f'{ents[i]}: init={[round(v,3) for v in init[i].tolist()]}')
    print(f'{pad}: lrnd={[round(v,3) for v in learned[i].tolist()]}')
    print(f'{pad}: diff={[round(v,3) for v in (learned[i]-init[i]).tolist()]}')
    print()
