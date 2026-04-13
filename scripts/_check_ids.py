import json
with open(r'C:\Users\LuisPlaza\MEMORIA\tucker_pull\notebooks\Experimento_warm_start\embeddings_8d_binario_s3\vocabulario_binario_8d.json') as f:
    v = json.load(f)
cursos = {'MA1101','MA1001','FI1000','BT1211','MA1002','MA1102','FI1100','CC1002','MA2001','MA2601','FI2001','FI2003','IQ2211'}
alumnos = [e for e in v['entities'] if e not in cursos]
print('Total entidades:', len(v['entities']))
print('Cursos:', len([e for e in v['entities'] if e in cursos]))
print('Alumnos:', len(alumnos))
print('Primeros 10:', alumnos[:10])
print('Ejemplo no-A:', [a for a in alumnos if not a.startswith('A')][:10])
