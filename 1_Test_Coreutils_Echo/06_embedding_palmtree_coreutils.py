"""
Script di Estrazione Embedding ibrido: Radare2 + Capstone + PalmTree.
Sviluppato per l'analisi di binari ottimizzati (-O0 vs -O3) nel dataset DeepBinDiff.

Il flusso di lavoro:
1. Usa Rabin2 (con output JSON) per una mappatura infallibile di nomi e indirizzi.
2. Usa Radare2 (r2pipe) per estrarre i byte esadecimali crudi senza formattazione.
3. Usa Capstone (con skipdata=True) per tradurre i byte in Assembly testuale, 
   ignorando i byte di allineamento/padding inseriti dal compilatore in -O3.
4. Usa il modello Transformer PalmTree per generare gli embedding vettoriali.
"""

import os
import sys
import json
import numpy as np
import subprocess
import re
import capstone
import torch
import r2pipe

# ==========================================
# PATCH PER PYTORCH (Retrocompatibilità)
# ==========================================
# I modelli pre-addestrati come PalmTree/SAFE usano vecchie versioni di PyTorch.
# Questa patch forza il caricamento dei tensori in CPU e disabilita il blocco di sicurezza
# 'weights_only' introdotto nelle nuove versioni di PyTorch, evitando eccezioni fatali.
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    kwargs['map_location'] = torch.device('cpu') 
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# Disabilitiamo CUDA forzando l'uso della CPU per compatibilità universale
import torch.nn as nn
nn.Module.cuda = lambda self, *args, **kwargs: self
torch.Tensor.cuda = lambda self, *args, **kwargs: self

# Aggiungiamo il percorso del modello linguistico PalmTree
sys.path.append(os.path.abspath('./PalmTree/pre-trained_model'))
import eval_utils as utils


def estrai_dati_con_rabin2(binary_path, target_name="ALL"):
    
    # Estrae la mappa delle funzioni dal binario utilizzando rabin2 in modalità JSON.
    # Vantaggio: L'output JSON garantisce la corretta estrazione dei nomi originali
    # delle funzioni anche nei binari "strippati" o pesantemente ottimizzati.
    
    print(f"[*] Cerco funzioni in {binary_path} con rabin2 JSON...")
    funzioni = []
    
    try:
        # Il parametro '-j' forza l'output strutturato, evitando errori di parsing testuale
        risultato = subprocess.check_output(['rabin2', '-s', '-j', binary_path], stderr=subprocess.STDOUT, text=True)
        dati_json = json.loads(risultato)
        
        # Gestione compatibilità versioni Radare2: 
        # Alcune versioni restituiscono un dict {"symbols": [...]}, altre direttamente una lista [...]
        if isinstance(dati_json, dict):
            simboli = dati_json.get('symbols', []) 
        else:
            simboli = dati_json 
            
        for sym in simboli:
            if isinstance(sym, dict) and sym.get('type') == 'FUNC':
                vaddr = sym.get('vaddr')
                size = sym.get('size', 0)
                name = sym.get('name', 'unknown')

                # Se l'utente cerca una funzione specifica (es. "main"), ignoriamo le altre
                if target_name != "ALL" and name != target_name and name != f"dbg.{target_name}" and name != f"sym.{target_name}":
                    continue

                # Ignoriamo le funzioni troppo piccole (es. stub del compilatore) per evitare rumore statistico
                if size < 10: 
                    continue

                funzioni.append({'vaddr': vaddr, 'size': size, 'name': name})
                
        print(f"[+] Trovate {len(funzioni)} funzioni da analizzare.")
        return funzioni
        
    except Exception as e:
        print(f"[!] Errore nell'analisi rabin2: {e}")
        return []


def estrai_vettori_palmtree_da_binario(binary_path, output_json, target_name="ALL"):
    
    # Funzione principale di orchestrazione. Coordina l'estrazione hex, 
    # la disassemblazione resiliente e l'interrogazione del Transformer.
    
    print(f"\n[*] Inizializzazione modello PalmTree per: {binary_path}")
    
    # Caricamento del vocabolario e dei pesi pre-addestrati del Transformer
    MODEL_PATH = "./PalmTree/pre-trained_model/palmtree/transformer.ep19"
    VOCAB_PATH = "./PalmTree/pre-trained_model/palmtree/vocab"
    palmtree = utils.UsableTransformer(model_path=MODEL_PATH, vocab_path=VOCAB_PATH)

    # Recupero la mappa delle funzioni
    lista_funzioni = estrai_dati_con_rabin2(binary_path, target_name)
    if not lista_funzioni:
        print("[!] Nessuna funzione valida trovata. Elaborazione interrotta.")
        return

    # Configurazione Disassemblatore Capstone
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    # skipdata=True permette di ignorare i byte non validi (es. NOP di padding, 
    # jump tables) inseriti aggressivamente dai compilatori in -O3, evitando il crash del disassemblatore.
    md.skipdata = True 

    # Manteniamo un'unica istanza di Radare2 aperta per velocizzare le letture di memoria
    print("[*] Apertura r2pipe per estrazione memoria raw...")
    r2 = r2pipe.open(binary_path, flags=['-2']) 
    r2.cmd('aaa') 

    risultati = []
    print("[*] Decodifica istruzioni e calcolo embedding vettoriali in corso...")
    
    # Disabilitiamo il calcolo dei gradienti di PyTorch per risparmiare memoria RAM
    with torch.no_grad():
        for i, func in enumerate(lista_funzioni):
            print(f"    -> [{i+1}/{len(lista_funzioni)}] Analizzo: {func['name']} ...", end=" ", flush=True)
            
            try:
                # Estrazione Byte Crudi: Spostamento all'indirizzo e dump esadecimale puro
                raw_output = r2.cmd(f's {func["vaddr"]}; p8 {func["size"]}')
                if not raw_output:
                    print("SKIP (Memoria inaccessibile)")
                    continue
                    
                # Pulizia della stringa da eventuali output testuali di errore di Radare2
                hex_pulito = re.sub(r'[^0-9a-fA-F]', '', raw_output)
                
                # Prevenzione crash Capstone: fromhex richiede un numero pari di caratteri
                if len(hex_pulito) % 2 != 0:
                    hex_pulito = hex_pulito[:-1]
                if not hex_pulito:
                    print("SKIP (Hex vuoto post-pulizia)")
                    continue
                    
                asm_bytes = bytes.fromhex(hex_pulito)
                
                # Traduzione in testo (Linguaggio di input per PalmTree)
                istruzioni_testo = []
                for inst in md.disasm(asm_bytes, func["vaddr"]):
                    # Ignoriamo i dati saltati grazie a skipdata (classificati come .byte precedentemente da md.skipdata = True)
                    if inst.mnemonic != ".byte":
                        istruzioni_testo.append(f"{inst.mnemonic} {inst.op_str}")
                
                if not istruzioni_testo:
                    print("SKIP (Nessuna istruzione Assembly valida estratta)")
                    continue
                
                # Tronchiamo le funzioni a un massimo di 512 istruzioni (come nel paper originale, limite a livello di istruzioni e non di sub-token) per prevenire out-of-memory o errori del tensore
                if len(istruzioni_testo) > 512:
                    istruzioni_testo = istruzioni_testo[:512]
                
                # Interrogazione del Transformer (genera un embedding per ciascuna istruzione)
                embeddings_istruzioni = palmtree.encode(istruzioni_testo)

                #Verifica se "embeddings_istruzioni" è un oggetto di tipo "Tensore" (la struttura dati usata da PyTorch)
                if torch.is_tensor(embeddings_istruzioni):
                    #Il comando .cpu() prende i dati dalla GPU e li alloca nella memoria RAM 
                    #Se i dati erano già nella CPU, questo comando non fa nulla e passa avanti.
                    embeddings_istruzioni = embeddings_istruzioni.cpu().numpy()
                    
                # Mean Pooling: calcoliamo la media dei vettori di ogni singola istruzione 
                # per ottenere un unico vettore (embedding) rappresentativo dell'intera macro-funzione
                vettore_funzione = np.mean(embeddings_istruzioni, axis=0)

                risultati.append({
                    "id_funzione": func["name"],
                    "vettore_palmtree": vettore_funzione.tolist() # La chiave 
                })
                
                print(f"OK ({len(istruzioni_testo)} istruz.)")
                
            except Exception as e:
                print(f"ERRORE di elaborazione neurale: {e}")

    r2.quit()

    # Salvataggio su disco del dataset generato
    if risultati:
        with open(output_json, "w") as f:
            json.dump(risultati, f, indent=4)
        print(f"\n[+] Operazione completata con successo. Generati {len(risultati)} vettori.")
        print(f"[*] Risultati serializzati in: {output_json}")


if __name__ == "__main__":
    # Esecuzione del test sull'utility Coreutils "echo"
    
    # Estrazione del vettore target (La funzione main dal binario non ottimizzato)
    BINARIO_O0 = "dataset_DeepBinDiff/coreutils/binaries/coreutils-5.93-O0/echo" 
    estrai_vettori_palmtree_da_binario(BINARIO_O0, "echo_main_O0_palmtree.json", target_name="main")

    # Estrazione dello spazio dei vettori (Tutte le funzioni dal binario ottimizzato)
    BINARIO_O3 = "dataset_DeepBinDiff/coreutils/binaries/coreutils-5.93-O3/echo" 
    estrai_vettori_palmtree_da_binario(BINARIO_O3, "echo_intero_O3_palmtree.json", target_name="ALL")