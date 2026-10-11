import os
import json
import csv
import numpy as np
from numpy.linalg import norm

def calcola_similarita(v1, v2):
    v1_np = np.array(v1)
    v2_np = np.array(v2)
    if norm(v1_np) == 0 or norm(v2_np) == 0:
        return 0.0
    return np.dot(v1_np, v2_np) / (norm(v1_np) * norm(v2_np))

def get_vector_and_model(filepath):
    try:
        with open(filepath, 'r') as f:
            dati = json.load(f)
            elem = dati[0]
            if "vettore_palmtree" in elem:
                return np.array(elem["vettore_palmtree"]), "PalmTree", elem["id_funzione"]
            elif "vettore_safe" in elem:
                return np.array(elem["vettore_safe"]), "SAFE", elem["id_funzione"]
            elif "vettore_jtrans" in elem:
                return np.array(elem["vettore_jtrans"]), "jTrans", elem["id_funzione"]
    except Exception:
        pass
    return None, None, None

def esegui_ranking(file_target, file_dataset, top_k=5):
    try:
        v_target, mod_target, id_target = get_vector_and_model(file_target)
        if v_target is None: return []
        
        with open(file_dataset, 'r') as f:
            dataset_candidati = json.load(f)
            
        risultati = []
        for cand in dataset_candidati:
            cand_id = cand["id_funzione"]
            if "vettore_palmtree" in cand:
                v_cand = cand["vettore_palmtree"]
            elif "vettore_safe" in cand:
                v_cand = cand["vettore_safe"]
            elif "vettore_jtrans" in cand:
                v_cand = cand["vettore_jtrans"]
            else:
                continue
                
            sim = calcola_similarita(v_target, v_cand)
            risultati.append((cand_id, sim * 100))
            
        risultati.sort(key=lambda x: x[1], reverse=True)
        return risultati[:top_k]
    except:
        return []

MODELS = ["safe", "palmtree", "jtrans"]
COMBINATIONS = [("O0", "O0"), ("O3", "O3"), ("O0", "O3"), ("O3", "O0")]

os.makedirs("tabelle_risultati", exist_ok=True)

for o_base, o_target in COMBINATIONS:
    csv_path = f"tabelle_risultati/{o_base}_vs_{o_target}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["Ranking", "Funzione (SAFE)", "% SAFE", "Delta SAFE", "Funzione (PalmTree)", "% PalmTree", "Delta PalmTree", "Funzione (jTrans)", "% jTrans", "Delta jTrans"])
        
        # Raccogliamo i ranking Top-5 per ciascun modello
        rankings_by_model = {}
        for mod in MODELS:
            t_file = f"echo_main_{o_base}_{mod}.json"
            d_file = f"echo_intero_{o_target}_{mod}.json"
            rankings_by_model[mod] = esegui_ranking(t_file, d_file, top_k=5)
            
        # Scriviamo le 5 righe
        for i in range(5):
            row = [f"#{i+1}"]
            for mod in MODELS:
                ranks = rankings_by_model[mod]
                if i < len(ranks):
                    cand_id, score = ranks[i]
                    row.append(cand_id)
                    row.append(f"{score:.2f}%")
                    if i == 0:
                        row.append("-")
                    else:
                        score_first = ranks[0][1]
                        delta = score_first - score
                        row.append(f"{delta:.2f}%")
                else:
                    row.append("N/A")
                    row.append("N/A")
                    row.append("N/A")
            writer.writerow(row)

print("Tabelle Ranking generate con successo in 'tabelle_risultati/'.")
