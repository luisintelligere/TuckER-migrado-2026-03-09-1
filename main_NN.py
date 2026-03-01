from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model_NN import TuckER  # Asegúrate de que model.py tenga la clase TuckER modificada
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import pandas as pd
import os

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

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data: er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for i, pair in enumerate(batch): targets[i, er_vocab[pair]] = 1.
        return np.array(batch), torch.FloatTensor(targets).to(self.device)

    # -----------------------------------------------------------------------
    # MÉTODO DE CARGA HÍBRIDA (MODIFICA AQUÍ LA LÓGICA DE TU CSV)
    # -----------------------------------------------------------------------
    def load_entity_features_hybrid(self, entities_list, feature_file="vector_notas.csv"):
        print(f"🔹 Cargando features desde {feature_file}...")
        
        # --- LÓGICA REAL (Descomentar y adaptar) ---
        # df = pd.read_csv(feature_file, index_col=0, sep=';') 
        # feature_dim = df.shape[1]
        
        # --- LÓGICA DUMMY (Para que el script corra sin el CSV) ---
        feature_dim = 4 # Supongamos 4 notas
        print("⚠️ USANDO FEATURES ALEATORIAS Y CRITERIO DUMMY PARA ALUMNOS (Modificar en código)")

        student_indices = []
        course_indices = []
        student_feats_list = []
        
        for i, ent_name in enumerate(entities_list):
            # 1. Identificar si es Alumno
            # Ejemplo Real: if ent_name in df.index:
            
            # Criterio Dummy: Si empieza con dígito (ej: '2019...') o 'A' es alumno
            is_student = ent_name[0].isdigit() or ent_name.startswith('A')
            
            if is_student:
                student_indices.append(i)
                
                # Ejemplo Real: feat = df.loc[ent_name].values
                feat = np.random.randn(feature_dim) # Dummy random
                student_feats_list.append(feat)
            else:
                course_indices.append(i)

        # Convertir a tensores
        student_idxs = torch.tensor(student_indices, dtype=torch.long).to(self.device)
        course_idxs = torch.tensor(course_indices, dtype=torch.long).to(self.device)
        student_features = torch.tensor(np.array(student_feats_list), dtype=torch.float).to(self.device)
        
        print(f"✅ Detectados: {len(student_idxs)} Alumnos (Red Neuronal) | {len(course_idxs)} Cursos (Embedding Estático)")
        return student_features, feature_dim, student_idxs, course_idxs

    def evaluate(self, model, data, student_features):
        hits = [[] for _ in range(10)]; ranks = []
        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))
        
        if not test_data_idxs:
            return {'hits@10': np.nan, 'hits@3': np.nan, 'hits@1': np.nan, 'mr': np.nan, 'mrr': np.nan}

        # Evaluar por batches
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0]).to(self.device)
            r_idx = torch.tensor(data_batch[:,1]).to(self.device)
            e2_idx = torch.tensor(data_batch[:,2]).to(self.device)
            
            # PASO CRÍTICO: Pasar features al forward
            predictions = model.forward(e1_idx, r_idx, student_features)
            
            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                if filt: predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value
            
            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()
            
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)
                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        metrics = {'hits@10': np.mean(hits[9]), 'hits@3': np.mean(hits[2]), 'hits@1': np.mean(hits[0]), 'mr': np.mean(ranks), 'mrr': np.mean(1./np.array(ranks))}
        # print(f"Eval stats: MRR {metrics['mrr']:.4f}")
        return metrics

    def train_and_eval(self, init_vocab_path=None, feature_file="vector_notas.csv"):
        print("🚀 Iniciando entrenamiento del modelo Híbrido (TuckER + NN)...")
        total_start_time = time.time()
        
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        train_data_idxs = self.get_data_idxs(d.train_data)
        
        # 1. Cargar Datos Híbridos
        student_features, feature_dim, student_idxs, course_idxs = \
            self.load_entity_features_hybrid(d.entities, feature_file)

        # 2. Instanciar Modelo Híbrido
        # Nota: model.py TuckER debe aceptar feature_dim, student_idxs, course_idxs en __init__
        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, feature_dim, 
                       student_idxs, course_idxs, **self.kwargs)
        model.to(self.device)
        model.init()

        # El optimizador verá todos los parámetros (NN + TuckER)
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate and self.decay_rate != 1.0:
            scheduler = ExponentialLR(opt, self.decay_rate)
        
        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        
        best_val_mrr = 0.0
        patience_counter = 0
        best_epoch = None
        early_stopping_epoch = None
        model_path = os.path.join(self.output_dir, "best_model.pt")
        os.makedirs(self.output_dir, exist_ok=True)

        print("Starting training...")
        for it in range(1, self.num_iterations + 1):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                
                e1_idx = torch.tensor(data_batch[:,0]).to(self.device)
                r_idx = torch.tensor(data_batch[:,1]).to(self.device)
                
                # 3. Forward con Features
                predictions = model.forward(e1_idx, r_idx, student_features)
                
                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))           
                
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            
            if self.decay_rate and self.decay_rate != 1.0:
                scheduler.step()
            
            # Logging
            epoch_loss = np.mean(losses)
            # print(f"Epoch: {it}, Loss: {epoch_loss:.4f}", end="\r")
            
            # Guardar histórico simplificado
            epoch_results = {'epoch': it, 'loss': epoch_loss}

            # 4. Evaluación Periódica (cada 1 época para chequear early stopping, o cada 10 para velocidad)
            # Evaluar siempre para early stopping correcto
            model.eval()
            with torch.no_grad():
                # Validación
                val_metrics = self.evaluate(model, d.valid_data, student_features)
                current_val_mrr = val_metrics['mrr']
                epoch_results['val_mrr'] = current_val_mrr
                
                print(f"Epoch {it} | Loss: {epoch_loss:.4f} | Val MRR: {current_val_mrr:.4f} | Time: {time.time()-start_train:.2f}s")

                if current_val_mrr > best_val_mrr:
                    # print(f"  -> New Best Model! (prev: {best_val_mrr:.4f})")
                    best_val_mrr = current_val_mrr
                    best_epoch = it
                    torch.save(model.state_dict(), model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if self.patience > 0 and patience_counter >= self.patience:
                    print(f"\n⏹️ Early stopping triggered at epoch {it}.")
                    early_stopping_epoch = it
                    break 

            self.history.append(epoch_results)
        
        # Test Final con el mejor modelo guardado
        if best_epoch:
            print(f"\nCargando mejor modelo (Epoch {best_epoch})...")
            model.load_state_dict(torch.load(model_path))
        
        model.eval()
        with torch.no_grad():
            print("Evaluando en Test Set...")
            test_metrics = self.evaluate(model, d.test_data, student_features)
            print("-" * 30)
            print(f"RESULTADOS FINALES (TEST):")
            print(f"MRR     : {test_metrics['mrr']:.4f}")
            print(f"Hits@1  : {test_metrics['hits@1']:.4f}")
            print(f"Hits@3  : {test_metrics['hits@3']:.4f}")
            print(f"Hits@10 : {test_metrics['hits@10']:.4f}")
            print("-" * 30)

        total_end_time = time.time()
        total_training_time = total_end_time - total_start_time
        print(f"⏱️ Tiempo total: {total_training_time:.2f} segundos")

        # Guardar métricas
        metrics_path = os.path.join(self.output_dir, "training_metrics.csv")
        pd.DataFrame(self.history).to_csv(metrics_path, index=False)
        print(f"✅ Métricas guardadas en {metrics_path}")

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
    parser.add_argument("--output_prefix", type=str, default="hybrid_experiment", nargs="?")
    parser.add_argument("--init_vocab", type=str, default=None)
    parser.add_argument("--patience", type=int, default=10, help="Paciencia para early stopping.")
    parser.add_argument("--feature_file", type=str, default="vector_notas.csv", help="Archivo con features de alumnos")
    
    args = parser.parse_args()
    data_dir = f"data/{args.dataset}/"
    output_dir = os.path.join("results", args.output_prefix)

    # Reproducibilidad
    torch.backends.cudnn.deterministic = True 
    seed = 20
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed) 
    
    d = Data(data_dir=data_dir, reverse=True)
    
    experiment = Experiment(
        learning_rate=args.lr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim,
        num_iterations=args.num_iterations, batch_size=args.batch_size, decay_rate=args.dr,
        cuda=args.cuda, input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1,
        hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing,
        output_dir=output_dir, patience=args.patience
    )
        
    experiment.train_and_eval(
        init_vocab_path=args.init_vocab,
        feature_file=args.feature_file
    )