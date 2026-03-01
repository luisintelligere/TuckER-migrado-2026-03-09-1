import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

class EmbeddingPredictor(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(EmbeddingPredictor, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)

class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, feature_dim, student_idxs, course_idxs, **kwargs):
        """
        student_idxs: Tensor con los índices globales de las entidades que son alumnos.
        course_idxs: Tensor con los índices globales de las entidades que son cursos.
        """
        super(TuckER, self).__init__()

        self.d1 = d1
        # Guardamos los índices como buffers (no son parámetros entrenables)
        self.register_buffer("student_idxs", student_idxs)
        self.register_buffer("course_idxs", course_idxs)

        # 1. Red Neuronal SOLO para Alumnos
        self.student_net = EmbeddingPredictor(feature_dim, d1)

        # 2. Embeddings Estáticos SOLO para Cursos
        num_courses = len(course_idxs)
        self.course_E = torch.nn.Embedding(num_courses, d1)

        # 3. Embeddings de Relaciones
        self.R = torch.nn.Embedding(len(d.relations), d2)
        
        # 4. Tensor Núcleo W
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                    dtype=torch.float, device="cpu", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

    def init(self):
        xavier_normal_(self.R.weight.data)
        xavier_normal_(self.course_E.weight.data)

    def forward(self, e1_idx, r_idx, student_features):
        """
        student_features: Tensor (Num_Alumnos x Feat_Dim) solo con notas de alumnos.
        """
        
        # === PASO A: Construcción de la Matriz E Híbrida ===
        
        # 1. Generar embeddings de alumnos con la NN
        student_embeddings = self.student_net(student_features) # Forma: (N_alumnos, d1)
        
        # 2. Obtener embeddings estáticos de cursos
        # Usamos todos los cursos para armar la matriz completa
        # (course_E es pequeño, solo tiene filas para cursos)
        course_embeddings = self.course_E.weight # Forma: (N_cursos, d1)

        # 3. Ensamblar la matriz E total (N_entidades, d1)
        # Creamos un contenedor vacío
        total_entities = self.student_idxs.size(0) + self.course_idxs.size(0)
        E = torch.zeros(total_entities, self.d1, device=e1_idx.device)
        
        # Colocamos cada grupo en su lugar correspondiente usando los índices globales
        E.index_copy_(0, self.student_idxs, student_embeddings)
        E.index_copy_(0, self.course_idxs, course_embeddings)

        # === PASO B: Lógica TuckER Standard con la matriz E ensamblada ===
        
        e1 = E[e1_idx] # Extraemos embeddings del batch (ya sean alumnos o cursos)
        
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        
        # Producto contra TODAS las entidades (Matriz E híbrida completa)
        x = torch.mm(x, E.transpose(1,0))
        
        pred = torch.sigmoid(x)
        return pred