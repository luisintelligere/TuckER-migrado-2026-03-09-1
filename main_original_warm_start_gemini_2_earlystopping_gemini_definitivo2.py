from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import pandas as pd
import os
import json

# Versión del codigo que ocupa una función de evaluación que rankea todas las relaciones incluso las *_reverse


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

    def evaluate(self, model, data, rank_relations=False, include_reverse=True, restrict_rel_names=None):
        """
        Evalúa el modelo en 'data'.

        Parámetros
        ----------
        rank_relations : bool
            False -> entity-ranking (como el original).
            True  -> relation-ranking (rankea relaciones para un (h,t) fijo).
        include_reverse : bool
            Si rank_relations=True, indica si se incluyen también relaciones con sufijo "_reverse".
        restrict_rel_names : list[str] | None
            Si se pasa una lista, solo se rankean esas relaciones (y, si include_reverse=True,
            también sus variantes *_reverse si existen en el vocabulario). Si es None, usa todas
            las permitidas por include_reverse.

        Retorna
        -------
        dict con: {'hits@10','hits@3','hits@1','mr','mrr'}
        """
        import numpy as np
        model.eval()

        # Índices (h,r,t) para el split objetivo
        test_data_idxs = self.get_data_idxs(data)
        if not test_data_idxs:
            return {'hits@10': np.nan, 'hits@3': np.nan, 'hits@1': np.nan, 'mr': np.nan, 'mrr': np.nan}

        # ====================== MODO ENTITY-RANKING (original) ======================
        if not rank_relations:
            hits = [[] for _ in range(10)]
            ranks = []

            # filtered setting sobre ENTIDADES: (h,r) -> {t verdaderos} en TODOS los splits
            er_vocab_all = self.get_er_vocab(self.get_data_idxs(d.data))
            print("Number of data points: %d" % len(test_data_idxs))

            for i in range(0, len(test_data_idxs), self.batch_size):
                batch = np.array(test_data_idxs[i:i+self.batch_size])
                e1_idx = torch.tensor(batch[:,0]).to(self.device)
                r_idx  = torch.tensor(batch[:,1]).to(self.device)
                e2_idx = torch.tensor(batch[:,2]).to(self.device)

                # scores para TODAS las entidades (tails)
                predictions = model.forward(e1_idx, r_idx)

                # filtra otros verdaderos excepto el gold
                for j in range(batch.shape[0]):
                    filt = er_vocab_all.get((batch[j,0], batch[j,1]), [])
                    target_value = predictions[j, e2_idx[j]].item()
                    if filt:
                        predictions[j, filt] = -1e9
                    predictions[j, e2_idx[j]] = target_value

                order = torch.argsort(predictions, dim=1, descending=True).cpu().numpy()
                gold  = e2_idx.cpu().numpy()
                for j in range(order.shape[0]):
                    rank = int(np.where(order[j] == gold[j])[0][0]) + 1
                    ranks.append(rank)
                    for k in range(10):
                        hits[k].append(1.0 if rank <= (k+1) else 0.0)

            ranks = np.asarray(ranks, dtype=float)
            metrics = {
                'hits@10': float(np.mean(hits[9])) if ranks.size else np.nan,
                'hits@3' : float(np.mean(hits[2])) if ranks.size else np.nan,
                'hits@1' : float(np.mean(hits[0])) if ranks.size else np.nan,
                'mr'     : float(np.mean(ranks))   if ranks.size else np.nan,
                'mrr'    : float(np.mean(1.0 / ranks)) if ranks.size else np.nan
            }
            print('Hits @10: {0}'.format(metrics['hits@10']))
            print('Hits @3: {0}'.format(metrics['hits@3']))
            print('Hits @1: {0}'.format(metrics['hits@1']))
            print('Mean rank: {0}'.format(metrics['mr']))
            print('Mean reciprocal rank: {0}'.format(metrics['mrr']))
            return metrics

        # ====================== MODO RELATION-RANKING (nuevo) ======================
        # 1) Definir el conjunto de relaciones candidatas a rankear
        if restrict_rel_names is not None:
            base_set = set(restrict_rel_names)
            if include_reverse:
                # añade *_reverse si existen en el vocab
                base_set |= {r + "_reverse" for r in restrict_rel_names if (r + "_reverse") in d.relations}
            rel_names = [r for r in d.relations if r in base_set]
        else:
            if include_reverse:
                rel_names = list(d.relations)  # TODAS las relaciones (incluye _reverse)
            else:
                rel_names = [r for r in d.relations if not r.endswith("_reverse")]

        # Mapea nombres a índices (usando los mapas de ESTA clase)
        rel_idxs = []
        for r in rel_names:
            if r not in self.relation_idxs:
                continue
            rel_idxs.append(self.relation_idxs[r])
        # Asegurar orden consistente (rel_names[i] ↔ rel_idxs[i])
        rel_pairs = list(zip(rel_names, rel_idxs))

        # 2) filtered setting sobre RELACIONES: (h,t) -> {r verdaderas} en TODOS los splits
        pair2rels = defaultdict(set)
        all_trip_idxs = self.get_data_idxs(d.data)  # train+valid+test
        for (h_i, r_i, t_i) in all_trip_idxs:
            pair2rels[(h_i, t_i)].add(r_i)

        hits = [[] for _ in range(10)]
        ranks = []

        print("Number of data points: %d" % len(test_data_idxs))
        # Recorremos con acceso a nombres crudos (para gold exacto)
        for (h_idx, r_idx, t_idx), (_, r_raw, _) in zip(test_data_idxs, data):
            # Si no incluimos reverse y la relación es *_reverse, saltamos este caso
            if (not include_reverse) and r_raw.endswith("_reverse"):
                continue

            e_h = model.E.weight[h_idx]
            e_t = model.E.weight[t_idx]

            # 3) Scores para todas las relaciones candidatas
            scores = []
            for _, ridx in rel_pairs:
                e_r = model.R.weight[ridx]
                # Contraer núcleo con r -> M_r (d_e x d_e)
                if model.W.shape[0] == e_r.shape[0]:
                    M_r = torch.tensordot(model.W, e_r, dims=([0],[0]))
                else:
                    M_r = torch.tensordot(model.W, e_r, dims=([1],[0]))
                s = (e_h @ M_r @ e_t).item()
                scores.append(s)
            scores = np.asarray(scores, dtype=float)

            # 4) Filtered setting en RELACIONES: anula otras verdaderas del mismo (h,t)
            true_rel_set = pair2rels.get((h_idx, t_idx), set())
            if true_rel_set:
                for j, (_, ridx) in enumerate(rel_pairs):
                    if ridx in true_rel_set and ridx != r_idx:
                        scores[j] = -1e9  # anula otras verdaderas distintas del gold

            # 5) Rank del GOLD (exactamente r_raw tal cual esté en rel_names)
            if r_raw not in rel_names:
                # si no está entre candidatas, no evaluamos este caso
                continue
            gold_pos = rel_names.index(r_raw)

            order = np.argsort(-scores)  # descendente
            rank = int(np.where(order == gold_pos)[0][0]) + 1

            ranks.append(rank)
            for k in range(10):
                hits[k].append(1.0 if rank <= (k+1) else 0.0)

        ranks = np.asarray(ranks, dtype=float)
        if ranks.size == 0:
            return {'hits@10': np.nan, 'hits@3': np.nan, 'hits@1': np.nan, 'mr': np.nan, 'mrr': np.nan}

        metrics = {
            'hits@10': float(np.mean(hits[9])),
            'hits@3' : float(np.mean(hits[2])),
            'hits@1' : float(np.mean(hits[0])),
            'mr'     : float(np.mean(ranks)),
            'mrr'    : float(np.mean(1.0 / ranks)),
        }
        print("\nEvaluación completada (relaciones):")
        print("Hits@1={:.4f} | Hits@2={:.4f} | MR={:.4f} | MRR={:.4f}".format(
            metrics['hits@1'], float(np.mean(hits[1])), metrics['mr'], metrics['mrr']
        ))
        return metrics




    def train_and_eval(self, init_embeddings_path=None, init_vocab_path=None):
        print("Training the TuckER model...")
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        train_data_idxs = self.get_data_idxs(d.train_data)
        
        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        model.to(self.device)

        if init_embeddings_path and init_vocab_path:
            print(f"🔹 Cargando embeddings iniciales desde {init_embeddings_path}")
            try:
                init_embeddings = torch.load(init_embeddings_path, map_location=self.device)
                with torch.no_grad():
                    model.E.weight.data.copy_(init_embeddings)
                print("✅ Embeddings iniciales inyectados en el modelo.")
            except Exception as e:
                print(f"⚠️ Error al cargar embeddings: {e}. Usando inicialización aleatoria.")
                model.init()
        else:
            model.init()

        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate and self.decay_rate != 1.0:
            scheduler = ExponentialLR(opt, self.decay_rate)
        
        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        
        alumnos_muestra = [e for e in d.entities if e.startswith("A") or e.isdigit()][:10]
        print(f"Se monitorearán los embeddings de los siguientes alumnos: {alumnos_muestra}")
        
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
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))           
                loss = model.loss(predictions, targets)
                loss.backward(); opt.step()
                losses.append(loss.item())
            if self.decay_rate and self.decay_rate != 1.0:
                scheduler.step()
            
            epoch_results = {'epoch': it, 'loss': np.mean(losses)}
            print(f"\nEpoch: {it}, Time: {time.time()-start_train:.4f}, Loss: {epoch_results['loss']:.4f}")
            
            model.eval()
            with torch.no_grad():
                emb_snapshot = {}
                for alumno in alumnos_muestra:
                    if alumno in self.entity_idxs:
                        idx = self.entity_idxs[alumno]
                        emb_snapshot[alumno] = model.E.weight[idx].cpu().numpy().tolist()
                emb_snapshot['epoch'] = it
                self.embeddings_history.append(emb_snapshot)

                print("Validation:")
                val_metrics = self.evaluate(model, d.valid_data, rank_relations=True, include_reverse=True)
                for key, value in val_metrics.items():
                    epoch_results['val_' + key] = value
                
                current_val_mrr = val_metrics['mrr']
                if current_val_mrr > best_val_mrr:
                    print(f"Validation MRR improved from {best_val_mrr:.4f} to {current_val_mrr:.4f}. Saving model...")
                    best_val_mrr = current_val_mrr
                    best_epoch = it
                    torch.save(model.state_dict(), model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"Validation MRR did not improve. Patience: {patience_counter}/{self.patience}")
                
                if self.patience > 0 and patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {it}.")
                    early_stopping_epoch = it
                    break 

                if not it % 2:
                    print("Test:")
                    test_metrics = self.evaluate(model, d.test_data)
                    for key, value in test_metrics.items():
                        epoch_results['test_' + key] = value
            
            self.history.append(epoch_results)
        
        if early_stopping_epoch is not None:
            if best_epoch is not None:
                print(f"\nEntrenamiento detenido por early stopping en la época {early_stopping_epoch}. El mejor modelo proviene de la época {best_epoch} (MRR = {best_val_mrr:.4f}).")
            else:
                print(f"\nEntrenamiento detenido por early stopping en la época {early_stopping_epoch}. No se registró una mejora en MRR para guardar un modelo.")
        else:
            if best_epoch is not None:
                print(f"\nEntrenamiento completado sin early stopping. El mejor modelo proviene de la época {best_epoch} (MRR = {best_val_mrr:.4f}).")
            else:
                print("\nEntrenamiento completado sin early stopping y sin mejoras registradas en MRR.")

        print("\nEntrenamiento finalizado. Guardando historiales...")
        metrics_path = os.path.join(self.output_dir, "training_metrics.csv")
        emb_path = os.path.join(self.output_dir, "embeddings_history.csv")
        pd.DataFrame(self.history).to_csv(metrics_path, index=False)
        pd.DataFrame(self.embeddings_history).to_csv(emb_path, index=False)
        print(f"✅ Métricas guardadas en {metrics_path}")
        print(f"✅ Historial de Embeddings guardado en {emb_path}")

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
    parser.add_argument("--output_prefix", type=str, default="my_experiment", nargs="?")
    parser.add_argument("--init_embeddings", type=str, default=None)
    parser.add_argument("--init_vocab", type=str, default=None)
    parser.add_argument("--patience", type=int, default=10, nargs="?", help="Paciencia para early stopping.")
    
    args = parser.parse_args()
    data_dir = f"data/{args.dataset}/"
    output_dir = os.path.join("results", args.output_prefix)

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
        init_embeddings_path=args.init_embeddings, 
        init_vocab_path=args.init_vocab
    )