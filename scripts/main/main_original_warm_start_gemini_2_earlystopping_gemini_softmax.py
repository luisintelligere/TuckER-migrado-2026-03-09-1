import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model_soft import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import pandas as pd
import os
import json

class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False,
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0., output_dir='results', patience=10):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.output_dir = output_dir
        self.patience = patience
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}
        self.device = torch.device("cuda" if (cuda and torch.cuda.is_available()) else "cpu")
        self.history = []
        self.embeddings_history = []

    def get_so_vocab(self, data):
        """
        Crea un diccionario mapping (subject, object) -> relation_index.
        Asume que para un par (s, o) existe una única relación verdadera (nota).
        """
        so_vocab = {}
        for triple in data:
            s_idx = self.entity_idxs[triple[0]]
            r_idx = self.relation_idxs[triple[1]]
            o_idx = self.entity_idxs[triple[2]]
            
            # Guardamos la relación como el target (clase)
            so_vocab[(s_idx, o_idx)] = r_idx
        return so_vocab

    def get_batch(self, so_vocab_keys, so_vocab, idx):
        """
        Genera un batch de pares (s, o) y sus targets (r).
        """
        batch_keys = so_vocab_keys[idx:idx+self.batch_size]
        
        batch_s = []
        batch_o = []
        batch_r = []
        
        for (s, o) in batch_keys:
            batch_s.append(s)
            batch_o.append(o)
            batch_r.append(so_vocab[(s, o)])
            
        # Convertimos a tensores
        # Inputs: (Batch,)
        # Targets: (Batch,) indices de clase para CrossEntropy
        return (torch.LongTensor(batch_s).to(self.device), 
                torch.LongTensor(batch_o).to(self.device), 
                torch.LongTensor(batch_r).to(self.device))

    def evaluate(self, model, data):
        """
        Evalúa Accuracy en la predicción de la relación (nota).
        """
        # Preparar datos de test
        so_vocab = self.get_so_vocab(data)
        so_keys = list(so_vocab.keys())
        
        if not so_keys:
            return {'accuracy': np.nan, 'loss': np.nan}

        print("Number of test examples: %d" % len(so_keys))
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(so_keys), self.batch_size):
                e1_idx, e2_idx, target_r = self.get_batch(so_keys, so_vocab, i)
                
                # Forward: Obtener logits de relaciones
                logits = model.forward(e1_idx, e2_idx)
                
                # Calcular Loss (solo para referencia)
                loss = model.loss(logits, target_r)
                total_loss += loss.item() * len(e1_idx)
                
                # Predicción: Argmax sobre los logits
                predictions = torch.argmax(logits, dim=1)
                
                # Accuracy
                correct = (predictions == target_r).sum().item()
                total_correct += correct
                total_samples += len(e1_idx)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        print(f'Test Loss: {avg_loss:.4f}')
        print(f'Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})')
        
        # MRR en este contexto (Ranking de relaciones correctas)
        # Si quisieras Hits@K sobre relaciones, podrías agregarlo aquí, 
        # pero Accuracy es la métrica reina para Softmax.
        
        return {'accuracy': accuracy, 'loss': avg_loss}

    def train_and_eval(self, init_embeddings_path=None, init_vocab_path=None):
        print("Training the TuckER (Relation Classification / Softmax) model...")
        
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        
        # Vocabulario de entrenamiento: (s, o) -> r
        so_vocab = self.get_so_vocab(d.train_data)
        so_vocab_keys = list(so_vocab.keys())
        
        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        model.to(self.device)
        
        if init_embeddings_path:
            # Lógica de carga de embeddings (simplificada para el ejemplo)
            print(f"Cargando embeddings pre-entrenados...")
            try:
                state = torch.load(init_embeddings_path, map_location=self.device)
                model.load_state_dict(state, strict=False) 
            except:
                model.init()
        else:
            model.init()

        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate and self.decay_rate != 1.0:
            scheduler = ExponentialLR(opt, self.decay_rate)
        
        # Monitoreo
        alumnos_muestra = [e for e in d.entities if e.startswith("A")][:5]
        
        best_val_acc = 0.0
        patience_counter = 0
        model_path = os.path.join(self.output_dir, "best_model_softmax.pt")
        os.makedirs(self.output_dir, exist_ok=True)

        print("Starting training loop...")
        for it in range(1, self.num_iterations + 1):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(so_vocab_keys)
            
            for j in range(0, len(so_vocab_keys), self.batch_size):
                e1_idx, e2_idx, target_r = self.get_batch(so_vocab_keys, so_vocab, j)
                
                opt.zero_grad()
                
                logits = model.forward(e1_idx, e2_idx)
                
                # CrossEntropyLoss espera logits y targets (indices de clase)
                loss = model.loss(logits, target_r)
                
                loss.backward()
                opt.step()
                losses.append(loss.item())
            
            if self.decay_rate and self.decay_rate != 1.0:
                scheduler.step()
            
            epoch_results = {'epoch': it, 'loss': np.mean(losses)}
            print(f"\nEpoch: {it}, Time: {time.time()-start_train:.4f}, Loss: {epoch_results['loss']:.4f}")
            
            # Validación
            model.eval()
            print("Validation:")
            val_metrics = self.evaluate(model, d.valid_data)
            epoch_results.update({f'val_{k}': v for k, v in val_metrics.items()})
            
            current_val_acc = val_metrics['accuracy']
            
            if current_val_acc > best_val_acc:
                print(f"Validation Accuracy improved: {best_val_acc:.4f} -> {current_val_acc:.4f}")
                best_val_acc = current_val_acc
                torch.save(model.state_dict(), model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{self.patience}")
            
            if self.patience > 0 and patience_counter >= self.patience:
                print("Early stopping triggered.")
                break 

            self.history.append(epoch_results)
        
        print("\nEntrenamiento finalizado.")
        pd.DataFrame(self.history).to_csv(os.path.join(self.output_dir, "metrics_softmax.csv"), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?")
    parser.add_argument("--edim", type=int, default=200, nargs="?")
    parser.add_argument("--rdim", type=int, default=200, nargs="?")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?")
    parser.add_argument("--output_prefix", type=str, default="softmax_experiment", nargs="?")
    parser.add_argument("--init_embeddings", type=str, default=None)
    
    # --- CORRECCIÓN AQUÍ: Agregamos el argumento faltante ---
    parser.add_argument("--init_vocab", type=str, default=None)
    # --------------------------------------------------------
    
    parser.add_argument("--patience", type=int, default=10, nargs="?")
    
    args = parser.parse_args()
    data_dir = f"data/{args.dataset}/"
    output_dir = os.path.join("results", args.output_prefix)

    torch.backends.cudnn.deterministic = True 
    seed = 20
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed) 
    
    # RECUERDA: reverse=False para la versión Softmax (Notas exactas)
    d = Data(data_dir=data_dir, reverse=False)
    
    experiment = Experiment(
        learning_rate=args.lr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim,
        num_iterations=args.num_iterations, batch_size=args.batch_size, decay_rate=args.dr,
        cuda=args.cuda, input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1,
        hidden_dropout2=args.hidden_dropout2, output_dir=output_dir, patience=args.patience
    )
    
    # --- CORRECCIÓN AQUÍ: Pasamos el argumento a la función ---
    experiment.train_and_eval(
        init_embeddings_path=args.init_embeddings, 
        init_vocab_path=args.init_vocab
    )
    # ----------------------------------------------------------