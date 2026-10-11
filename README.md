# 🔬 Comparative Evaluation of Neural Models for Binary Code Similarity Detection (BCSD)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Disassemblers](https://img.shields.io/badge/Tools-Radare2%20%7C%20IDA%20Pro%20%7C%20Capstone-blueviolet.svg)](https://radare.org/)
[![Architectures](https://img.shields.io/badge/Target%20Arch-x86--64%20%7C%20RISC--V-green.svg)](https://en.wikipedia.org/wiki/RISC-V)
[![Thesis](https://img.shields.io/badge/Thesis-M.Sc.%20Computer%20Engineering-red.svg)](https://www.unicas.it/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Official Experimental Repository for the Master's Thesis in Computer Engineering:**  
> *"Valutazione Comparativa di Modelli Neurali per la Binary Code Similarity Detection"*  
> **Candidate:** Vincenzo Minchella ([vincenzo.sporting@gmail.com](mailto:vincenzo.sporting@gmail.com))  
> **Supervisors:** Prof.ssa Alessandra Scotto di Freca, Prof. Francesco Fontanella  
> **Institution:** Università degli Studi di Cassino e del Lazio Meridionale (DIEI) — Academic Year 2025/2026

---

## 📌 Executive Summary & Scientific Motivation

**Binary Code Similarity Detection (BCSD)** is crucial for vulnerability discovery, software supply-chain auditing, malware analysis, and patch identification without source code. In recent years, Deep Learning architectures treating disassembled machine instructions as natural language (**NLP-based BCSD**) have largely replaced traditional graph isomorphism and heuristic matching.

However, mainstream literature often suffers from an **evaluation bias**:
1. Most existing benchmarks evaluate queries almost exclusively at unoptimized compilation levels (`-O0`), obscuring how models behave when target queries are compiled with aggressive optimizations (`-O3`).
2. High-level metric summaries frequently proclaim a universal superiority of Transformer-based models over sequential recurrent networks, ignoring subtle architectural trade-offs, fine-grained semantic blind spots, and the structural failure modes of Control-Flow Graphs (CFG).

This repository presents an **objective, rigorous, and balanced comparative evaluation** of three representative neural paradigms:
* **SAFE** (*Self-Attentive Function Embeddings*): Bi-directional GRU with self-attentive pooling (Massarelli et al., DIMVA 2019).
* **PalmTree**: Assembly language model (BERT-base) using per-instruction tokenization and global Mean Pooling (Li et al., CCS 2021).
* **jTrans**: Jump-Aware Transformer integrating token sequences with CFG relative jump distance positional encodings (Wang et al., ISSTA 2022).

```
+---------------------------------------------------------------------------------------------------------+
|                                    NEURAL BCSD BENCHMARK PIPELINE                                       |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Target Binaries (x86-64 / RISC-V)            Disassembly & Preprocessing           Neural Models      |
|   +-------------------------------+            +---------------------------+       +------------------+ |
|   | GNU Coreutils: echo (v5.93)   | ---------> | Radare2 / Capstone (NLP)  | ----> | SAFE (BiGRU)     | |
|   | Buffer Overflow (CWE-120/121) |            +---------------------------+       +------------------+ |
|   | Compilations: -O0 vs -O3      | ---------> | IDA Pro (CFG Extraction)  | ----> | PalmTree (BERT)  | |
|   +-------------------------------+            +---------------------------+       +------------------+ |
|                                                                                    | jTrans (CFG-TF)  | |
|                                                                                    +------------------+ |
|                                                                                             |           |
|                                                                                             v           |
|   Evaluation & Benchmark Analytics <------------------------------------------ Cosine Similarity Space  |
|   * Testbed 1: 4-Way Cross-Optimization Search Space Ranking & Delta Separation                         |
|   * Testbed 2: Fine-Grained Vulnerability Triangulation (Vuln vs Safe vs Incognito)                     |
|   * Testbed 3: Cross-Architecture Generalization (x86-64 vs RISC-V)                                     |
+---------------------------------------------------------------------------------------------------------+
```

---

## 🔬 Architectural Comparison & Computational Trade-offs

| Model | Underlying Architecture | Input Representation | Max Sequence Length | Pooling Strategy | CFG Aware? | Disassembler Dependency | Inference Cost & Overhead |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **SAFE** | Bi-directional GRU + Self-Attention | Normalized instructions (Opcode + operands) | 150 instructions (zero-padded) | Self-Attentive Weighted Sum | ❌ No (Sequential) | **Open-source** (Radare2 / Capstone) | **Very Low** (Near-instantaneous) |
| **PalmTree**| 12-layer Transformer (BERT-base) | Sub-instruction assembly tokens | 20 tokens / instruction | Global Mean Pooling over all instructions | ⚠️ Weak (Pretrained on CFG/DFG) | **Open-source** (Capstone) | **Medium** (Per-instruction BERT) |
| **jTrans** | 12-layer Transformer + Jump-Aware Positional Encoding | Assembly tokens + Basic Block Jump Encodings | 512 tokens / function | `[CLS]` token embedding | ✅ Yes (Full CFG) | **Proprietary** (**IDA Pro** required) | **High** (IDA batch scripting + TF) |

---

## 🧪 Experimental Testbeds

### Testbed 1: Semantic Ranking (1-to-N Search Space on GNU Coreutils `echo` v5.93)
* **Dataset:** GNU Coreutils `echo` (v5.93) from the standard academic benchmark **DeepBinDiff**.
* **Task:** Retrieve the target function (`main`) within a candidate search space across all 4 combinatorial optimization settings:
  1. **`-O0` vs `-O0`**: Homogeneous unoptimized baseline.
  2. **`-O0` vs `-O3`**: Forward Cross-Optimization (unoptimized query vs optimized search pool).
  3. **`-O3` vs `-O0`**: Reverse Cross-Optimization (optimized query vs unoptimized search pool).
  4. **`-O3` vs `-O3`**: Homogeneous optimized baseline.
* **Evaluation Metric:** **$\Delta$ Separation Margin** ($\text{Sim}(\text{Rank 1}) - \text{Sim}(\text{Rank 2})$), quantifying how decisively the model isolates the ground truth from runner-up false positives.

### Testbed 2: Semantic Vulnerability Discovery (Fine-Grained Triangulation)
* **Task:** Determine whether models identify **true semantic danger** or are fooled by **syntactic/topological disguise**.
* **Triangulation Triplet:**
  * **`vuln_A`**: Vulnerable reference function using unbounded `strcpy` (CWE-120 Buffer Overflow).
  * **`safe_A`**: Secure patched reference function using bounded `strncpy`.
  * **`vuln_incognito`**: Syntactically disguised target implementing a manual pointer-copy loop (`while (*src) { *dst++ = *src++; }`). Structurally, it introduces a loop and branch edges that differ from `vuln_A`, while remaining **semantically vulnerable**.
* **Triangulation Margin:**
  $$\Delta_{\text{vuln}} = \text{Sim}(\text{Incognito}, \text{Vuln}) - \text{Sim}(\text{Incognito}, \text{Safe})$$
  * $\Delta_{\text{vuln}} > 0$: **Correct classification** (Incognito correctly recognized as closer to the vulnerable implementation).
  * $\Delta_{\text{vuln}} < 0$: **Critical Security Failure / False Negative** (Incognito incorrectly classified as safe!).

---

## 📊 Comprehensive Experimental Results

### 1. Testbed 1: Semantic Ranking (Coreutils `echo`)

The empirical data proves that **there is no absolute winner**. Rather, the performance hierarchy **inverts strictly based on the compilation level of the query function**:

#### Summary: $\Delta$ Separation Margins Across All 4 Scenarios
| Scenario | Query Level | Search Pool Level | SAFE $\Delta$ | PalmTree $\Delta$ | jTrans $\Delta$ | Winner |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`-O0` vs `-O0`** | `-O0` | `-O0` | 21.14% | 2.98% | **28.12%** | 🏆 **jTrans** |
| **`-O0` vs `-O3`** | `-O0` | `-O3` | 3.12% | 1.20% | **21.92%** | 🏆 **jTrans** |
| **`-O3` vs `-O0`** | `-O3` | `-O0` | **16.20%** | 0.16% | 13.07% | 🏆 **SAFE** |
| **`-O3` vs `-O3`** | `-O3` | `-O3` | **33.40%** | 1.63% | 24.45% | 🏆 **SAFE** |

#### Detailed Top-5 Ranking Tables

##### Scenario 1: `-O0` vs `-O0` (Homogeneous Unoptimized)
| Rank | SAFE Function | Sim | $\Delta$ | PalmTree Function | Sim | $\Delta$ | jTrans Function | Sim | $\Delta$ |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :--- | :---: | :---: |
| **#1** | **`main`** | **100.00%** | — | **`main`** | **100.00%** | — | **`main`** | **100.00%** | — |
| **#2** | `close_stdout` | 78.86% | **21.14%** | `quotearg_buffer_restyled` | 97.02% | **2.98%** | `usage` | 71.88% | **28.12%** |
| **#3** | `usage` | 70.56% | 29.44% | `close_stdout` | 95.29% | 4.71% | `close_stdout` | 57.66% | 42.34% |
| **#4** | `quotearg_buffer_restyled`| 55.50% | 44.50% | `parse_long_options` | 93.81% | 6.19% | `version_etc_va` | 55.36% | 44.64% |
| **#5** | `quotearg_n_options` | 48.98% | 51.02% | `x2nrealloc_inline` | 92.80% | 7.20% | `xalloc_die` | 51.41% | 48.59% |

##### Scenario 2: `-O0` vs `-O3` (Forward Cross-Optimization)
| Rank | SAFE Function | Sim | $\Delta$ | PalmTree Function | Sim | $\Delta$ | jTrans Function | Sim | $\Delta$ |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :--- | :---: | :---: |
| **#1** | **`main`** | **69.18%** | — | **`main`** | **96.03%** | — | **`main`** | **80.16%** | — |
| **#2** | `usage` | 66.06% | **3.12%** | `quotearg_buffer_restyled` | 94.83% | **1.20%** | `usage` | 58.24% | **21.92%** |
| **#3** | `quotearg_buffer_restyled`| 50.76% | 18.41% | `parse_long_options` | 90.69% | 5.34% | `quotearg_style` | 53.32% | 26.84% |
| **#4** | `quotearg_char` | 50.26% | 18.91% | `x2nrealloc` | 89.99% | 6.04% | `close_stdout` | 50.62% | 29.54% |
| **#5** | `close_stdout` | 49.23% | 19.94% | `version_etc_va` | 89.81% | 6.22% | `quotearg_colon` | 45.54% | 34.62% |

##### Scenario 3: `-O3` vs `-O0` (Reverse Cross-Optimization)
| Rank | SAFE Function | Sim | $\Delta$ | PalmTree Function | Sim | $\Delta$ | jTrans Function | Sim | $\Delta$ |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :--- | :---: | :---: |
| **#1** | **`main`** | **69.18%** | — | **`main`** | **96.03%** | — | **`main`** | **80.16%** | — |
| **#2** | `close_stdout` | 52.98% | **16.20%** | `close_stdout` | 95.87% | **0.16%** | `usage` | 67.09% | **13.07%** |
| **#3** | `version_etc_va` | 49.59% | 19.59% | `hextobin` | 93.38% | 2.65% | `__libc_csu_init` | 58.10% | 22.05% |
| **#4** | `usage` | 47.90% | 21.28% | `quotearg_buffer_restyled` | 92.61% | 3.42% | `clone_quoting_options` | 50.95% | 29.21% |
| **#5** | `quotearg_n_options` | 47.53% | 21.65% | `xnmalloc_inline` | 91.34% | 4.69% | `close_stdout` | 48.74% | 31.42% |

##### Scenario 4: `-O3` vs `-O3` (Homogeneous Optimized)
| Rank | SAFE Function | Sim | $\Delta$ | PalmTree Function | Sim | $\Delta$ | jTrans Function | Sim | $\Delta$ |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :--- | :---: | :---: |
| **#1** | **`main`** | **100.00%** | — | **`main`** | **100.00%** | — | **`main`** | **100.00%** | — |
| **#2** | `version_etc_va` | 66.60% | **33.40%** | `quotearg_buffer_restyled` | 98.37% | **1.63%** | `usage` | 75.55% | **24.45%** |
| **#3** | `usage` | 62.92% | 37.08% | `version_etc_va` | 96.67% | 3.33% | `quotearg_style` | 65.75% | 34.25% |
| **#4** | `quotearg_char` | 57.80% | 42.20% | `x2nrealloc` | 93.58% | 6.42% | `close_stdout` | 63.88% | 36.12% |
| **#5** | `quotearg_colon` | 52.79% | 47.21% | `parse_long_options` | 93.37% | 6.63% | `quotearg_n` | 63.26% | 36.74% |

#### 🔬 Scientific Discussion of Testbed 1:
1. **Query at `-O0` (jTrans Dominance):** When the query function is compiled without optimization, its Control-Flow Graph is completely preserved. In this setting, jTrans's jump-aware positional encoding breaks the vector saturation of purely textual models, yielding substantial separation margins ($\Delta = 28.12\%$ and $21.92\%$). This confirms existing literature (e.g., UniASM).
2. **Query at `-O3` (The Revival of SAFE):** In reverse cross-optimization (`-O3` vs `-O0`) and homogeneous optimized (`-O3` vs `-O3`), **SAFE achieves the widest separation gaps in the entire experimental campaign ($\Delta = 16.20\%$ and $33.40\%$)**, surpassing jTrans ($13.07\%$ and $24.45\%$).
   * *Mechanism:* GCC `-O3` applies aggressive function inlining, loop unrolling, and instruction rescheduling that fundamentally alter or shatter the CFG. Because jTrans couples token embeddings with jump coordinates, structural disruptions in the query heavily degrade its latent representation. In contrast, SAFE's sequential Bi-GRU with self-attention relies exclusively on normalized instruction streams, rendering it resilient to topological perturbations.
   * *Original Contribution:* Most academic benchmarks test queries compiled at `-O0`. Demonstrating SAFE's superiority under `-O3` query regimes fills a major gap in the state of the art.
3. **PalmTree's Structural Saturation:** PalmTree never wins any scenario, producing $\Delta$ margins consistently below $3\%$ (and as low as $0.16\%$). Its instruction-level Mean Pooling compresses embeddings into an overly dense hypercone, severely limiting its discriminative capacity.

---

### 2. Testbed 2: Semantic Vulnerability Discovery (Fine-Grained Triangulation)

Evaluating whether models distinguish true semantic danger (`vuln_A` vs `safe_A`) when an unknown target (`vuln_incognito`) implements the vulnerability via syntactic disguise (manual `while` pointer loop):

| Compilation Regime | Model | Incognito vs Vuln (I-V) | Incognito vs Safe (I-S) | Margin $\Delta_{\text{vuln}}$ | Classification Outcome |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`-O0` vs `-O0`** | **SAFE** | **82.59%** | **73.88%** | **+8.71%** | ✅ **Correct (Best in Class)** |
| | **jTrans** | 81.35% | 85.05% | **-3.70%** | ❌ **CRITICAL FAILURE (False Negative)** |
| | **PalmTree** | 94.80% | 94.67% | **+0.13%** | ⚠️ Indifferent (Noise level) |
| **`-O0` vs `-O3` (Cross)** | **SAFE** | **38.09%** | **30.92%** | **+7.17%** | ✅ **Correct (Winner)** |
| | **jTrans** | 51.94% | 51.09% | **+0.85%** | ⚠️ Correct, but near noise floor |
| | **PalmTree** | 82.91% | 83.28% | **-0.37%** | ❌ False Negative (Negligible delta) |
| **`-O3` vs `-O0` (Cross)** | **SAFE** | **55.49%** | **50.41%** | **+5.08%** | ✅ **Correct (Winner)** |
| | **jTrans** | 74.24% | 74.02% | **+0.22%** | ⚠️ Correct, but near noise floor |
| | **PalmTree** | 88.47% | 89.30% | **-0.83%** | ❌ False Negative (Negligible delta) |
| **`-O3` vs `-O3`** | **SAFE** | 39.89% | 44.08% | **-4.19%** | ❌ False Negative |
| | **jTrans** | 74.36% | 74.04% | **+0.32%** | ⚠️ Correct, but microscopic margin |
| | **PalmTree** | 86.14% | 86.04% | **+0.10%** | ⚠️ Indifferent (Noise level) |

#### 🔍 Structural & Theoretical Analysis (CFG Inspection):
1. **The Isomorphic 3-Block CFG Blind Spot in jTrans:**
   * In source code, `vuln_A` (`strcpy`) and `safe_A` (`strncpy`) differ by only a single boundary parameter inside the function body. Consequently, their **Control Flow Graphs are completely identical (isomorphic 3-block graphs)**: Entry/Prologue $\rightarrow$ Function Call $\rightarrow$ Exit/Epilogue.
   * Conversely, `vuln_incognito` implements a `while` loop, introducing **additional basic blocks, branch conditions, and loop back-edges**.
   * At `-O0`, jTrans relies heavily on CFG topology. Because `vuln_A` and `safe_A` have identical graphs, jTrans finds no structural signal to distinguish the security patch. Concurrently, the graph distance introduced by the loop in `vuln_incognito` causes jTrans to suffer from **topological bias**, mistakenly classifying the vulnerable code closer to the safe reference (**False Negative at $\Delta = -3.70\%$**).
2. **SAFE's Precision in Intra-Block Semantic Triage:**
   * Operating directly on normalized instruction sequences without CFG constraints, SAFE successfully detects that the memory operations in `vuln_incognito` match the unsafe semantics of `vuln_A`. It achieves the strongest margin (**+8.71%**) at `-O0` and wins both cross-optimization regimes (**+7.17%** and **+5.08%**).
   * SAFE only fails at `-O3` vs `-O3` (-4.19%), where GCC unrolls the `while` loop into vectorized instructions, altering instruction lengths beyond SAFE's 150-token window.
3. **PalmTree's Complete Unreliability:**
   * PalmTree produces negligible margins ($< 0.15\%$) and fails in both cross-optimization directions ($-0.37\%$ and $-0.83\%$), proving that its dense latent space is blind to fine-grained vulnerability semantics.

---

### 3. Testbed 3: Cross-Architecture Generalization (x86-64 vs RISC-V)

Evaluating zero-shot transfer across radically different Instruction Set Architectures (x86-64 CISC vs RISC-V RISC):

| Comparison Pair (-O0 vs -O0) | SAFE Similarity | PalmTree Similarity | jTrans Similarity |
| :--- | :---: | :---: | :---: |
| **Safe vs Safe (S-S)** | 22.73% | **90.48%** | 29.82% |
| **Vulnerable vs Safe (V-S)** | 28.05% | **90.39%** | 30.36% |
| **Incognito vs Safe (I-S)** | 22.84% | **88.98%** | 32.56% |
| **Incognito vs Vulnerabile (I-V)** | 22.84% | **87.79%** | 33.03% |
| **Vulnerable vs Vulnerabile (V-V)** | 28.05% | **89.24%** | 30.10% |

* **Out-Of-Vocabulary (OOV) Collapse:** Both **SAFE** and **jTrans** experience total collapse (~22%–33%, equivalent to random noise). Their token dictionaries and embedding matrices are strictly bound to x86 assembly mnemonics and register names.
* **PalmTree's Generalization:** PalmTree retains ~88%–90% similarity, demonstrating that its sub-instruction tokenization scheme abstracts semantic concepts better across distinct instruction set architectures.

---

## 🎯 General Scientific Conclusions

1. **The Fundamental Limit of Cosine Similarity in Vulnerability Discovery:**
   * Current neural BCSD models are **not ready to be deployed as autonomous, reliable tools for fine-grained vulnerability discovery**.
   * In close-range security triage (distinguishing a 1-instruction patch or detecting disguised loops), separation margins collapse below 1%, entering the zone of decision uncertainty and statistical noise.
2. **Alignment with the Authors of jTrans (Wang et al., ISSTA 2022):**
   * As explicitly noted by Wang et al., neural BCSD architectures are designed and calibrated for **large-scale 1-to-N search and ranking across massive pools (thousands of candidate functions)**.
   * Applying these models to close-range discrimination over small function pools introduces a **severe task misalignment**: the model optimizes for coarse global similarity rather than fine-grained security semantics.
3. **The Computational & Operational Trade-off:**
   * **SAFE** offers an outstanding efficiency-to-accuracy ratio. It requires **no CFG parsing**, has **zero dependence on proprietary disassemblers (IDA Pro)**, runs with near-instantaneous inference, and outperforms jTrans under optimized queries (`-O3`) and intra-block instruction edits.
   * **jTrans** provides superior discriminative power only when the topological signal of the CFG is clear and undisturbed, but exhibits critical structural rigidity when graphs collide or are destructuring by aggressive compilers.

---

## 📂 Repository Structure

```
.
├── 2_Test_Coreutils_Echo/                    # Testbed 1: Semantic Ranking (1-to-N)
│   ├── dataset_DeepBinDiff/                 # Disassembled Coreutils echo v5.93 binaries
│   ├── tabelle_risultati/                   # Complete benchmark CSVs (O0_vs_O0, O0_vs_O3, O3_vs_O0, O3_vs_O3)
│   ├── 05_embedding_safe_coreutils.py       # SAFE feature extraction & embedding pipeline
│   ├── 06_embedding_palmtree_coreutils.py   # PalmTree inference pipeline (via Capstone)
│   ├── 07_embedding_jtrans_coreutils.py     # jTrans Transformer inference pipeline
│   ├── 08_ricerca_coreutils.py              # Top-K Cosine Similarity ranking engine
│   └── genera_tabelle_ranking.py            # Automated CSV benchmark exporter
│
├── 3_Vulnerability_Discovery/                # Testbed 2 & 3: Fine-Grained Triangulation & Cross-Arch
│   ├── dataset/                             # x86-64 C sources and compiled binaries
│   ├── datasetCompilatiArchitetturaDiversa/ # RISC-V cross-compiled binaries
│   ├── embeddingX86/                        # Extracted x86-64 JSON embedding vectors
│   ├── embeddingRiscv/                      # Extracted RISC-V JSON embedding vectors
│   ├── tabelle_risultati/                   # Intra-arch and cross-arch CSV benchmark results
│   ├── 09_embedding_safe_vuln.py            # SAFE vulnerability extraction
│   ├── 10_embedding_palmtree_vuln.py        # PalmTree vulnerability extraction
│   ├── 11_embedding_jtrans_vuln.py          # jTrans vulnerability extraction
│   ├── 12_ricerca_vuln.py                   # Triangular vulnerability evaluator
│   └── genera_tabelle_ranking.py            # Benchmark CSV generator
│
├── requirements.txt                         # Global Python dependencies
├── .gitignore                               # Strict filter blocking files >100MB, models, and virtualenvs
└── README.md                                # Project documentation
```

---

## ⚙️ Prerequisites & Environment Setup

### 1. System Requirements
* **OS:** Linux (Ubuntu 20.04/22.04, Kali Linux, or WSL2)
* **Conda:** Miniconda or Anaconda
* **Disassemblers:**
  * `radare2` / `rabin2` (v5.8.0+)
  * `IDA Pro` (7.5+ or 9.x) with Python 3 support (for jTrans CFG extraction)
* **Compilers:** `gcc` (x86-64) and `riscv64-linux-gnu-gcc` (for RISC-V cross-compilation)

### 2. Install Disassemblers & Compilers
```bash
sudo apt-get update
sudo apt-get install -y radare2 gcc-riscv64-linux-gnu build-essential git
```

### 3. Setup Python Environments

#### A. Main Environment (`base` or `tirocinio`)
```bash
conda activate base
pip install -r requirements.txt
```

#### B. Dedicated jTrans Environment (`jtrans`)
```bash
conda create -n jtrans python=3.8 pandas tqdm networkx pyelftools -y
conda activate jtrans
pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install simpletransformers
```

#### C. Downloading Pre-trained Model Weights
Since pre-trained checkpoints exceed GitHub's 100 MB file limit, download them directly from the authors' official releases:
* **PalmTree:** Download [`palmtree.tar.gz`](https://drive.google.com/file/d/1yC3M-kVTFWql6hCgM_QCbKtc1PbdVdvp/view?usp=sharing) and extract into `PalmTree/`.
* **jTrans:** Download [`models.tar.gz`](https://cloud.vul337.team:8443/s/tM5qGQPJa6iynCf) and [`experiments.tar.gz`](https://cloud.vul337.team:8443/s/wmqzYFyJnSEfEgm) and extract into `jTrans/`.
* **SAFEtorch:** Download `SAFEtorch.pt` from the official [SAFEtorch repository](https://github.com/gadiluna/SAFEtorch) into `SafeTorch/model/`.

#### D. IDA Pro Symlink Configuration
Link your local IDA Pro installation inside jTrans data utilities:
```bash
cd jTrans/datautils
ln -s /path/to/your/ida-pro-install ./ida
cd ../..
```

---

## 🚀 Step-by-Step Reproduction Guide

### Reproducing Testbed 1 (Coreutils Search Space Ranking)

```bash
cd 2_Test_Coreutils_Echo

# 1. Generate SAFE embeddings (Radare2 + RNN)
python3 05_embedding_safe_coreutils.py

# 2. Generate PalmTree embeddings (Capstone + BERT Mean Pooling)
python3 06_embedding_palmtree_coreutils.py

# 3. Generate jTrans CFG & Embeddings
conda deactivate
cd jTrans/datautils && python3 run.py && cd ../..
conda activate jtrans
python3 07_embedding_jtrans_coreutils.py

# 4. Evaluate Ranking (Top 10) in base environment
conda activate base
python3 08_ricerca_coreutils.py -t echo_main_O0_safe.json -d echo_intero_O3_safe.json -k 10
python3 08_ricerca_coreutils.py -t echo_main_O0_palmtree.json -d echo_intero_O3_palmtree.json -k 10
python3 08_ricerca_coreutils.py -t echo_main_O0_jtrans.json -d echo_intero_O3_jtrans.json -k 10

# 5. Export LaTeX / Excel CSV Tables
python3 genera_tabelle_ranking.py
```

---

### Reproducing Testbed 2 (Vulnerability Discovery & Triangulation)

```bash
cd 3_Vulnerability_Discovery

# 1. Feature Extraction (select dataset folder in scripts)
python3 09_embedding_safe_vuln.py
python3 10_embedding_palmtree_vuln.py

# jTrans IDA & Transformer pipeline
conda deactivate
cd jTrans/datautils && python3 run.py && cd ../..
conda activate jtrans
python3 11_embedding_jtrans_vuln.py

# 2. Run Triangular Evaluation
conda activate base
python3 12_ricerca_vuln.py -i vuln_incognito_O0_jtrans.json -v vuln_A_O0_jtrans.json -s safe_A_O0_jtrans.json
python3 12_ricerca_vuln.py -i vuln_incognito_O0_safe.json -v vuln_A_O0_safe.json -s safe_A_O0_safe.json

# 3. Export Benchmark Tables
python3 genera_tabelle_ranking.py
```

---

## 📖 Citation

If you use this benchmark, empirical findings, or methodology, please consider citing:

```bibtex
@mastersthesis{minchella2026bcsd,
  author       = {Vincenzo Minchella},
  title        = {Valutazione Comparativa di Modelli Neurali per la Binary Code Similarity Detection},
  school       = {Universit{\`a} degli Studi di Cassino e del Lazio Meridionale},
  department   = {Dipartimento di Ingegneria Elettrica e dell'Informazione "Maurizio Scarano"},
  year         = {2026},
  address      = {Cassino, Italy},
  note         = {Supervisors: Prof.ssa Alessandra Scotto di Freca, Prof. Francesco Fontanella}
}
```

### Foundational References
```bibtex
@inproceedings{massarelli2019safe,
  title={SAFE: Self-Attentive Function Embeddings for Binary Similarity},
  author={Massarelli, Luca and Di Luna, Giuseppe Antonio and Petroni, Fabio and Querzoni, Leonardo and Baldoni, Roberto},
  booktitle={Detection of Intrusions and Malware, and Vulnerability Assessment (DIMVA)},
  year={2019}
}

@inproceedings{li2021palmtree,
  title={PalmTree: Learning an Assembly Language Model for Instruction Embedding},
  author={Li, Xuezixiang and Qu, Yu and Yin, Heng},
  booktitle={Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security (CCS)},
  pages={3236--3251},
  year={2021}
}

@inproceedings{wang2022jtrans,
  title={jTrans: Jump-Aware Transformer for Binary Code Similarity Detection},
  author={Wang, Hao and Qu, Wenzheng and Katz, Gil and Zhu, Wenyu and Gao, Zeyu and Sawaby, Noah and Qi, Yan and Shen, Chao},
  booktitle={Proceedings of the 31st ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA)},
  pages={1--13},
  year={2022}
}
```

---

## 👤 Author & Contact

* **Author:** Vincenzo Minchella
* **Email:** [vincenzo.sporting@gmail.com](mailto:vincenzo.sporting@gmail.com)
* **Academic Institution:** [Università degli Studi di Cassino e del Lazio Meridionale](https://www.unicas.it/)
* **Department:** Dipartimento di Ingegneria Elettrica e dell'Informazione (DIEI)

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).
