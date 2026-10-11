"""
Script di Estrazione e Inferenza Neurale - Architettura jTrans
Sviluppato per la vettorizzazione semantica di interi binari (Dataset DeepBinDiff).

L'obiettivo di questo modulo è generare gli embedding necessari per un'analisi di 
similarità 1-a-N. Il flusso prevede l'estrazione di uno specifico vettore target 
(Query) da un binario non ottimizzato, e la successiva estrazione di tutte le 
funzioni da un binario altamente ottimizzato (Spazio di Ricerca), al fine di 
valutare il potere discriminante del modello Transformer CFG-Aware.
"""

import os
import sys
import json
import pickle
import torch
from transformers import BertTokenizer, BertModel

sys.path.append(os.path.abspath('./jTrans'))

# Importiamo direttamente lo script originale di jTrans
import data 

# L'archittettura di jTrans
class BinBertModel(BertModel):
    """
    Sovrascrittura del modello BERT standard. 
    A differenza del NLP tradizionale (che legge in modo strettamente sequenziale),
    qui i vettori di posizione (position_embeddings) vengono fusi con i vettori 
    delle parole (word_embeddings). Questo trucco architetturale permette alla rete 
    di mappare non solo "quali" istruzioni ci sono, ma "come" sono collegate 
    spazialmente tra loro nel Control Flow Graph estratto da IDA.
    """
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings.position_embeddings = self.embeddings.word_embeddings


def estrai_vettori_jtrans_da_binario(pkl_in, json_out, target_name="ALL"):
    """
    Scansiona un database serializzato (.pkl) generato da IDA, esegue 
    l'inferenza tramite jTrans e memorizza i vettori semantici in formato JSON.
    """
    print(f"\n[*] Analisi topologica e inferenza sul file: {pkl_in}")
    
    # Percorsi dei pesi pre-addestrati e del vocabolario
    MODEL_PATH = "./jTrans/models/jTrans-finetune"
    TOKENIZER_PATH = "./jTrans/jtrans_tokenizer"
    
    try:
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
        model = BinBertModel.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"[!] Errore nel caricamento del modello/tokenizer: {e}")
        return
    
    # model.eval() disabilita il Dropout e i layer di Normalizzazione, 
    # garantendo che lo stesso input produca sempre lo stesso output (determinismo)
    model.eval()
    
    # Allocazione tensori su acceleratore hardware (se presente)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Caricamento del file pickle estratto tramite lo script IDAPython (run.py)
    try:
        with open(pkl_in, 'rb') as f:
            dati_ida = pickle.load(f)
    except FileNotFoundError:
        print(f"[!] Errore: File {pkl_in} non trovato.")
        return

    risultati = []
    
    # Disabilitazione del grafo computazionale per il calcolo dei gradienti,
    # ottimizzando l'allocazione di memoria durante l'inferenza massiva.
    with torch.no_grad():
        for func_name, data_list in dati_ida.items():
            
            # Filtro per l'estrazione selettiva della Query Target
            if target_name != "ALL":
                # Gestione dei prefissi introdotti staticamente dai disassemblatori
                if func_name != target_name and func_name != f"sym.{target_name}" and func_name != f"dbg.{target_name}":
                    continue

            # data_list[1] contiene la sequenza lineare delle istruzioni Assembly
            asm_list = data_list[1]
            
            # Esclusione delle funzioni non significative (stub di compilazione)
            if not asm_list or len(asm_list) < 5: 
                continue
            
            # Passiamo l'intera lista di dati (che contiene il CFG) allo script originale.
            # Il parametro convert_jump=True innesca la conversione JUMP_ADDR_XXX.
            testo = data.gen_funcstr(data_list, convert_jump=True)
            
            # Tokenizzazione: Trasforma le stringhe in ID numerici comprensibili da BERT.
            # Limite architetturale: 512 token. Il padding riempie le funzioni corte, 
            # la truncation taglia quelle troppo lunghe.
            inputs = tokenizer([testo], add_special_tokens=True, max_length=512, 
                               padding='max_length', truncation=True, return_tensors='pt')
            
            # Esecuzione della Forward Propagation
            output = model(input_ids=inputs['input_ids'].to(device), 
                           attention_mask=inputs['attention_mask'].to(device))
            
            # Estrazione del pooler_output: 
            # È un layer finale di BERT che condensa l'intera matrice della funzione 
            # in un singolo vettore unidimensionale (generalmente di dimensione 768)
            vettore = output.pooler_output.flatten().cpu().numpy().tolist()
            
            risultati.append({
                "id_funzione": func_name, 
                "vettore_jtrans": vettore
            })
            
            # Feedback visivo per il monitoraggio dell'estrazione batch
            if len(risultati) % 10 == 0:
                print(f"    -> Vettorizzate {len(risultati)} funzioni nel dataset corrente...")

    # Serializzazione per la fase successiva di calcolo della Cosine Similarity
    if risultati:
        with open(json_out, "w") as f:
            json.dump(risultati, f, indent=4)
        print(f"[*] Operazione completata. Spazio vettoriale salvato in: {json_out} (Totale funzioni elaborabili: {len(risultati)})")
    else:
        print("[!] Nessuna funzione vettorizzata. Controlla i filtri o il file di input.")

if __name__ == "__main__":
    
    # I percorsi da cui prelevare i file strutturali (.pkl) generati da IDA 
    PKL_O0 = "./jTrans/datautils/extract/echo_O0_extract.pkl"
    PKL_O3 = "./jTrans/datautils/extract/echo_O3_extract.pkl"

    # 1. Costruzione della Query Semantica (Target).
    # Viene estratta esclusivamente la funzione bersaglio ('main') dal binario 
    # compilato in formato non ottimizzato (-O0).
    estrai_vettori_jtrans_da_binario(PKL_O0, "echo_main_O0_jtrans.json", target_name="main")
    
    # 2. Costruzione dello Spazio di Ricerca Multidimensionale.
    # Vengono estratte e vettorizzate tutte le funzioni valide presenti nel 
    # binario sottoposto a ottimizzazione aggressiva (-O3).
    estrai_vettori_jtrans_da_binario(PKL_O3, "echo_intero_O3_jtrans.json", target_name="ALL")