import torch
import numpy as np
from load_data import Data
from model import TuckER

def predecir(modelo, data_loader, cabeza, relacion, top_k=5):
    """
    Usa el modelo entrenado para predecir las 'colas' más probables para una
    'cabeza' y 'relación' dadas.
    """
    # Verificamos si la entidad y relación existen en los diccionarios
    if cabeza not in data_loader.entity_idxs:
        print(f"Error: La entidad cabeza '{cabeza}' no fue encontrada en los datos.")
        return []
    if relacion not in data_loader.relation_idxs:
        print(f"Error: La relación '{relacion}' no fue encontrada en los datos.")
        return []

    # Mapeamos los nombres a sus IDs numéricos
    cabeza_idx = torch.tensor([data_loader.entity_idxs[cabeza]])
    relacion_idx = torch.tensor([data_loader.relation_idxs[relacion]])

    # Ponemos el modelo en modo de evaluación (desactiva dropouts, etc.)
    modelo.eval()
    
    # Realizamos la predicción sin calcular gradientes
    with torch.no_grad():
        # model.forward devuelve las puntuaciones para TODAS las posibles entidades cola
        predicciones = modelo.forward(cabeza_idx, relacion_idx)
        
        # Obtenemos los índices de las 'k' puntuaciones más altas
        top_indices = torch.topk(predicciones, k=top_k).indices.tolist()[0]
        
        # Creamos un mapa inverso de ID a nombre de entidad para interpretar los resultados
        idx_a_entidad = {i: entidad for entidad, i in data_loader.entity_idxs.items()}
        
        # Convertimos los índices predichos a nombres de entidades
        resultados = [idx_a_entidad[idx] for idx in top_indices]
        
        return resultados

# --- PASO 1: Cargar los datos y el modelo entrenado ---

print("Cargando datos y mapeos...")
# Asegúrate de que la ruta a tu dataset sea la correcta
d = Data(data_dir="data/mis_datos_multirelacional/", reverse=True)

# **CORRECCIÓN 1: Crear los diccionarios de mapeo que faltaban**
# La clase Data solo crea las listas (.entities), no los diccionarios (.entity_idxs)
d.entity_idxs = {entity: i for i, entity in enumerate(d.entities)}
d.relation_idxs = {relation: i for i, relation in enumerate(d.relations)}

print("Definiendo la arquitectura del modelo...")
# **CORRECCIÓN 2: Usar la firma correcta del constructor de TuckER**
# Se pasan los argumentos posicionales (d, edim, rdim) y luego los kwargs
kwargs = {
    "input_dropout": 0.2,
    "hidden_dropout1": 0.2,
    "hidden_dropout2": 0.3
}
modelo = TuckER(d, 200, 200, **kwargs)

print("Cargando el modelo entrenado desde 'modelo_entrenado.pt'...")
# **CORRECCIÓN 3: Forzar la carga en CPU para evitar errores de CUDA**
modelo.load_state_dict(torch.load('modelo_entrenado.pt', map_location=torch.device('cpu')))

print("\n--- ¡Modelo listo para hacer predicciones! ---")


# --- PASO 2: Probar el modelo con ejemplos ---
# Cambia los valores de 'cabeza' y 'relacion' para hacer tus propias pruebas
try:
    # Seleccionamos una entidad real de tus datos para el ejemplo
    alumno_ejemplo = next(e for e in d.entities if e.startswith('A')) # Busca el primer alumno
    
    # Ejemplo 1: ¿Qué cursos es más probable que apruebe este alumno?
    predicciones_aprueba = predecir(modelo, d, cabeza=alumno_ejemplo, relacion='aprueba', top_k=5)
    print(f"\nPredicciones para '{alumno_ejemplo}' con relación 'aprueba': {predicciones_aprueba}")

    # Ejemplo 2: ¿Qué cursos es más probable que repruebe este alumno?
    predicciones_reprueba = predecir(modelo, d, cabeza=alumno_ejemplo, relacion='reprueba', top_k=5)
    print(f"Predicciones para '{alumno_ejemplo}' con relación 'reprueba': {predicciones_reprueba}")

except StopIteration:
    print("\nNo se encontraron entidades de alumnos en el dataset para hacer una predicción de ejemplo.")