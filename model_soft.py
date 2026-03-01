import numpy as np
import torch
from torch.nn.init import xavier_normal_

class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        
        # W shape: (d_r, d_e, d_e) -> (relation_dim, entity_dim, entity_dim)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                    dtype=torch.float, device="cpu", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        
        # CAMBIO: Usamos CrossEntropyLoss para clasificación multiclase (Softmax implícito)
        self.loss = torch.nn.CrossEntropyLoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        # CAMBIO: Este batchnorm ahora actúa sobre la dimensión de la relación (d2)
        # tras la contracción inicial.
        self.bn1 = torch.nn.BatchNorm1d(d2)
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, e2_idx):
        """
        Input: 
            e1_idx: Índices de Alumnos (Batch,)
            e2_idx: Índices de Cursos (Batch,)
        Output:
            logits: Scores para cada posible nota/relación (Batch, Num_Relations)
        """
        # 1. Embeddings de Sujeto y Objeto
        e1 = self.E(e1_idx)
        e2 = self.E(e2_idx)

        # 2. Regularización de entrada (Normalización y Dropout)
        e1 = self.bn0(e1)
        e1 = self.input_dropout(e1)
        
        # Aplicamos lo mismo al objeto (Curso) para simetría
        e2 = self.bn0(e2) 
        e2 = self.input_dropout(e2)

        # 3. Contracción Tensorial: (W x_2 e1 x_3 e2)
        # W: [d_r, d_e, d_e] (Relación, Sujeto, Objeto)
        # e1: [batch, d_e]
        # e2: [batch, d_e]
        # Resultado esperado -> [batch, d_r] (Vector latente de relación)
        
        # x[b, r] = Sum_{i, j} W[r, i, j] * e1[b, i] * e2[b, j]
        x = torch.einsum('rij,bi,bj->br', self.W, e1, e2)

        # 4. Procesamiento del vector latente
        x = self.bn1(x) # Normalizamos en el espacio de relaciones
        x = self.hidden_dropout1(x)

        # 5. Proyección al espacio de salida (Scores por cada relación existente)
        # x: [batch, d_r]
        # R: [num_relations, d_r]
        # Logits = x @ R.T -> [batch, num_relations]
        x = torch.mm(x, self.R.weight.transpose(1,0))
        
        # No aplicamos sigmoide, devolvemos logits puros para CrossEntropy
        return x