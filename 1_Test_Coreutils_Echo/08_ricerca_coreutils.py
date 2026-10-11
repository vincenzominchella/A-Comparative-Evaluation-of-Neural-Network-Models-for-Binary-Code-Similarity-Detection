"""
Script di Valutazione Top-K (Ranking Semantico).
Sviluppato per valutare la robustezza dei modelli di Machine Learning (SAFE/PalmTree) 
contro l'offuscamento indotto dalle ottimizzazioni del compilatore (-O0 vs -O3).

Metodologia:
1. Acquisizione della "Funzione Target" (Query vettoriale estratta dal binario in -O0).
2. Scansione dello "Spazio di Ricerca" (Tutte le funzioni estratte dal binario in -O3).
3. Calcolo della Cosine Similarity tra il Target e ogni candidato nello spazio di ricerca.
4. Generazione di un ranking Top-K per verificare in quale posizione l'architettura 
   neurale colloca la funzione semanticamente equivalente.
"""

import os
import sys
import json
import argparse
import numpy as np
from numpy.linalg import norm

def calcola_similarita(v1, v2):
    """
    Calcola la Similarità del Coseno tra due vettori (embedding).
    Formula: (v1 · v2) / (||v1|| * ||v2||)
    """
    v1_np = np.array(v1)
    v2_np = np.array(v2)
    
    # Prevenzione della divisione per zero in caso di vettori nulli (es. funzioni vuote)
    if norm(v1_np) == 0 or norm(v2_np) == 0:
        return 0.0
        
    return np.dot(v1_np, v2_np) / (norm(v1_np) * norm(v2_np))

def estrai_dati_vettore(elemento_json):
    """
    Ispeziona il dizionario JSON per determinare dinamicamente l'architettura 
    (SAFE, PalmTree o jTrans) utilizzata per la generazione dell'embedding.
    """
    if "vettore_palmtree" in elemento_json:
        return np.array(elemento_json["vettore_palmtree"]), "PalmTree"
    elif "vettore_safe" in elemento_json:
        return np.array(elemento_json["vettore_safe"]), "SAFE"
    elif "vettore_jtrans" in elemento_json:
        return np.array(elemento_json["vettore_jtrans"]), "jTrans"
    else:
        raise KeyError("Architettura non riconosciuta: chiave di embedding mancante.")

def esegui_ranking_semantico(file_target, file_spazio_ricerca, top_k=5):
    """
    Esegue il confronto 1-a-N e stampa la classifica dei risultati.
    """
    print(f"[*] Caricamento Funzione Target (Query -O0) da: {os.path.basename(file_target)}")
    try:
        with open(file_target, 'r') as f:
            dati_target = json.load(f)
            target_id = dati_target[0]["id_funzione"]
            vettore_target, modello_target = estrai_dati_vettore(dati_target[0])
    except Exception as e:
        print(f"[!] Errore nel caricamento del file Target: {e}")
        sys.exit(1)

    print(f"[*] Caricamento Spazio di Ricerca (Dataset -O3) da: {os.path.basename(file_spazio_ricerca)}")
    try:
        with open(file_spazio_ricerca, 'r') as f:
            dataset_candidati = json.load(f)
    except Exception as e:
        print(f"[!] Errore nel caricamento del file Dataset: {e}")
        sys.exit(1)

    totale_funzioni = len(dataset_candidati)
    if totale_funzioni == 0:
        print("[!] Il dataset di ricerca è vuoto. Interruzione.")
        sys.exit(1)

    # Controllo di coerenza architetturale
    _, modello_dataset = estrai_dati_vettore(dataset_candidati[0])
    if modello_target != modello_dataset:
        print(f"[!] ERRORE CRITICO: Incompatibilità vettoriale.")
        print(f"[!] Target generato con {modello_target}, Dataset generato con {modello_dataset}.")
        sys.exit(1)

    print(f"[*] Architettura rilevata: {modello_target}")
    print(f"[*] Avvio scansione 1-a-N su {totale_funzioni} candidati...\n")

    risultati = []
    
    # Valutazione della distanza semantica tra la Query e l'intero Spazio di Ricerca
    for candidato in dataset_candidati:
        candidato_id = candidato["id_funzione"]
        vettore_candidato, _ = estrai_dati_vettore(candidato)
        
        similarita = calcola_similarita(vettore_target, vettore_candidato)
        risultati.append((candidato_id, similarita))

    # Ordinamento decrescente (Classifica Top-K)
    risultati.sort(key=lambda x: x[1], reverse=True)

    # ==========================================
    # STAMPA FORMATTATA DEI RISULTATI
    # ==========================================
    print("="*65)
    print(f" VALUTAZIONE RANKING: Ricerca di '{target_id}'")
    print(f" MODELLO UTILIZZATO: {modello_target}")
    print("="*65)
    print(f"{'RANK':<5} | {'SIMILARITÀ':<12} | {'CANDIDATO (-O3)'}")
    print("-" * 65)
    
    for i in range(min(top_k, totale_funzioni)):
        candidato_id, score = risultati[i]
        percentuale = score * 100
        
        # Evidenziazione visiva del Rank #1
        if i == 0:
            print(f" #{i+1:<3} | {percentuale:>6.2f}%     | ---> {candidato_id}")
        else:
            print(f" #{i+1:<3} | {percentuale:>6.2f}%     | {candidato_id}")
    print("="*65 + "\n")

if __name__ == "__main__":
    # Configurazione argomenti da terminale
    parser = argparse.ArgumentParser(
        description="Esegue un ranking semantico tra una Funzione Target e un Dataset di funzioni."
    )
    parser.add_argument("-t", "--target", required=True, help="Percorso del file JSON contenente la Funzione Target (es. echo_main_O0.json)")
    parser.add_argument("-d", "--dataset", required=True, help="Percorso del file JSON contenente lo Spazio di Ricerca (es. echo_intero_O3.json)")
    parser.add_argument("-k", "--topk", type=int, default=5, help="Numero di risultati da mostrare in classifica (Default: 5)")
    
    args = parser.parse_args()
    
    esegui_ranking_semantico(args.target, args.dataset, top_k=args.topk)