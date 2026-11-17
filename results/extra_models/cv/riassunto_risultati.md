# WiC 2×2 ablation — riassunto risultati (COMPLETO al 17/11/2026, ore 16:00)

**Protocollo** (identico per tutti i backbone, come nel notebook `wic_ablation_study.ipynb`):
- Dataset: WiC (SuperGLUE), pool train+dev = **6066** esempi etichettati
- Validazione incrociata: **StratifiedKFold 5**, `CV_SEED=1234`, `set_seed(1234+fold)` per fold
- 6 condizioni: `full`, `no_highlight`, `no_prefix`, `plain` (2×2 highlight×prefisso) + `word_only`, `pos_only` (decomposizione del prefisso)
- Iperparametri: stessi Optuna del notebook (tunati su DeBERTa-large); FP32, batch effettivo 16, 4 epoche
- Metriche: **Acc OOF** (predizioni out-of-fold aggregate sul pool), **Acc media±std sui 5 fold**, **Δ vs `full`** in punti percentuali

**Stato del run al 17/11/2026 (ore 16:00)**: **210/210 run completati** (v4 alle 14:01, v5/UltraBERT alle 16:00). Tutti i 7 backbone a 30/30. Tabelle definitive generate con `--stats-only`; controverifica indipendente: acc OOF ricontate da zero sui pred salvati = valori del CSV (±0.00), 5 fold/pool senza sovrapposizioni, 210 chiavi = 42 combinazioni × 5 fold.

---

## Tabella principale (formato Tabella 4 del paper)

| Backbone | Condizione | Acc OOF | Acc fold (±std) | Δ vs full |
|---|---|---:|---:|---:|
| **deberta-large** *(notebook)* | full | 78.55 | 78.55 ± 0.76 | +0.00 |
| | no_highlight | 80.81 | 80.81 ± 1.58 | +2.26 |
| | no_prefix | 78.52 | 78.52 ± 1.70 | −0.03 |
| | plain | 79.29 | 79.29 ± 0.83 | +0.74 |
| | word_only | 79.67 | 79.67 ± 1.60 | +1.12 |
| | pos_only | 80.20 | 80.20 ± 1.36 | +1.65 |
| **roberta-large** *(notebook)* | full | 77.37 | 77.37 ± 1.21 | +0.00 |
| | no_highlight | 77.88 | 77.88 ± 1.14 | +0.51 |
| | no_prefix | 76.66 | 76.66 ± 1.01 | −0.71 |
| | plain | 75.70 | 75.70 ± 1.32 | −1.67 |
| | word_only | 77.53 | 77.53 ± 1.40 | +0.16 |
| | pos_only | 75.95 | 75.95 ± 1.19 | −1.42 |
| **nomic-bert-2048** 🆕 | full | **67.16** | 67.16 ± 1.67 | +0.00 |
| | no_highlight | 61.44 | 61.44 ± 4.03 | −5.72 |
| | no_prefix | 62.22 | 62.22 ± 3.54 | −4.95 |
| | plain | 59.05 | 59.05 ± 4.11 | −8.11 |
| | word_only | 61.64 | 61.64 ± 7.14 | −5.52 |
| | pos_only | 53.36 | 53.36 ± 3.15 | −13.80 |
| **ettin-encoder-150m** 🆕 | full | 68.71 | 68.71 ± 2.08 | +0.00 |
| | no_highlight | 70.85 | 70.85 ± 1.05 | +2.14 |
| | no_prefix | 68.41 | 68.41 ± 2.07 | −0.30 |
| | plain | **70.97** | 70.97 ± 1.15 | +2.26 |
| | word_only | 70.59 | 70.59 ± 1.91 | +1.88 |
| | pos_only | 70.44 | 70.44 ± 1.62 | +1.73 |
| **EuroBERT-210m** 🆕 | full | 72.62 | 72.62 ± 4.65 | +0.00 |
| | no_highlight | 72.60 | 72.60 ± 5.32 | −0.02 |
| | no_prefix | **73.99** | 73.99 ± 0.77 | +1.37 |
| | plain | 71.76 | 71.76 ± 4.81 | −0.86 |
| | word_only | 71.86 | 71.86 ± 5.53 | −0.76 |
| | pos_only | 73.56 | 73.56 ± 1.59 | +0.94 |
| **chandar-lab/NeoBERT** 🆕 | full | 76.10 | 76.10 ± 1.53 | +0.00 |
| | no_highlight | **77.12** | 77.12 ± 1.24 | +1.02 |
| | no_prefix | 74.22 | 74.22 ± 2.47 | −1.88 |
| | plain | 75.12 | 75.12 ± 0.44 | −0.97 |
| | word_only | 76.28 | 76.28 ± 0.84 | +0.18 |
| | pos_only | 76.11 | 76.11 ± 2.44 | +0.02 |
| **jhu-clsp/mmBERT-base** 🆕 | full | 74.17 | 74.17 ± 0.74 | +0.00 |
| | no_highlight | 74.68 | 74.68 ± 0.81 | +0.51 |
| | no_prefix | 74.81 | 74.81 ± 1.42 | +0.64 |
| | plain | 74.27 | 74.27 ± 0.70 | +0.10 |
| | word_only | 75.11 | 75.11 ± 0.56 | +0.94 |
| | pos_only | **75.22** | 75.22 ± 1.70 | +1.06 |
| **answerdotai/ModernBERT-base** 🆕 | full | 69.37 | 69.37 ± 2.11 | +0.00 |
| | no_highlight | 70.52 | 70.52 ± 1.08 | +1.15 |
| | no_prefix | 68.46 | 68.46 ± 1.55 | −0.91 |
| | plain | 70.49 | 70.49 ± 1.29 | +1.12 |
| | word_only | 70.06 | 70.06 ± 0.95 | +0.69 |
| | pos_only | **71.00** | 71.00 ± 1.22 | +1.63 |
| **dmis-lab/UltraBERT-150M** 🆕 | full | 73.46 | 73.46 ± 6.88 | +0.00 |
| | no_highlight | 78.17 | 78.17 ± 1.53 | +4.71 |
| | no_prefix | 72.62 | 72.62 ± 7.01 | −0.84 |
| | plain | 76.31 | 76.31 ± 1.63 | +2.85 |
| | word_only | 77.73 | 77.73 ± 0.78 | +4.27 |
| | pos_only | **78.34** | 78.34 ± 1.84 | +4.88 |

---

## Contrasti per backbone (effetti principali)

| Backbone | Δ highlight (full − no_hl) | Δ prefisso (full − no_prefix) | word_only − plain | pos_only − plain |
|---|---:|---:|---:|---:|
| deberta-large | −2.26 | +0.03 | +0.38 | +0.91 |
| roberta-large | +0.51 | +0.71 | +1.83 | +0.25 |
| **nomic-bert-2048** | **+5.72** | **+4.95** | +2.59 | **−5.69** |
| **ettin-encoder-150m** | −2.14 | +0.29 | −0.38 | −0.53 |
| **EuroBERT-210m** | −0.02 | −1.37 | +0.10 | +1.80 |
| **chandar-lab/NeoBERT** | −1.02 | +1.88 | +1.16 | +0.99 |
| **jhu-clsp/mmBERT-base** | −0.51 | −0.64 | +0.84 | +0.96 |
| **answerdotai/ModernBERT-base** | −1.15 | +0.91 | −0.43 | +0.51 |
| **dmis-lab/UltraBERT-150M** | **−4.71** | +0.84 | +1.42 | +2.03 |

*(Δ highlight positivo = togliere i token `<TW>` migliora; Δ prefisso positivo = togliere il prefisso peggiora.)*

---

## Letture chiave

1. **5/7 backbone: i marcatori espliciti sono neutri o dannosi.** DeBERTa-lg, RoBERTa-lg, Ettin, EuroBERT, NeoBERT, mmBERT e ModernBERT (5 dei 7 nuovi + i 2 del notebook) mostrano Δ highlight ≤ +1.2 pp e Δ prefisso di piccola magnitudine: per un modello con capacità sufficiente, i token `<TW>` e il prefisso non aggiungono informazione (e spesso la sottraggono, da rumore).
2. **NomicBERT e UltraBERT sono le due eccezioni marcate**: per entrambi `full`/`no_highlight` sono le condizioni migliori e i marcatori espliciti sono utili.
   - NomicBERT: Δ highlight +5.72, Δ prefisso +4.95; `pos_only` crolla a 53.36 (−13.80 vs full), l'effetto più estremo della tabella.
   - UltraBERT: Δ highlight −4.71 (togliere `<TW>` **migliora** di 4.71 pp, p_Holm 1.4e-15) e Δ prefisso +0.84; `pos_only` è la sua migliore condizione (78.34, +4.88 vs full, p_Holm 6.4e-05). Contrariamente a NomicBERT, il suo guadagno viene dalla **posizione** (pos_component +2.03, word_component +1.42, entrambi significativi), non dal highlight.
   - I due modelli "dipendenti dai marcatori" li usano in modo **opposto**: NomicBERT dal highlight (`<TW>`), UltraBERT dalla posizione.
3. **A parità quasi di dimensione, effetti opposti**: Ettin (150M) segue il trend "large", NomicBERT (137M) no. La variabile spiegativa più probabile non è la capacità ma **pretraining/tokenizer** (NomicBERT: BPE 256k, pretraining orientato a retrieval).
4. **La decomposizione del prefisso non è universale**:
   - NomicBERT: la *parola* aiuta (+2.59), la *posizione* fa male (−5.69)
   - EuroBERT: la *posizione* aiuta (+1.80), la *parola* no (+0.10)
   - NeoBERT, RoBERTa e mmBERT: entrambe le componenti aiutano lievemente (mmBERT: +0.84 parola, +0.96 posizione, nessuna significativa)
   - UltraBERT: entrambe le componenti utili e **significative** (parola +1.42 p_Holm 0.009, posizione +2.03 p_Holm 6.4e-05) — il solo modello oltre a NomicBERT per cui la decomposizione del prefisso conta davvero
   - ModernBERT: nessuna componente utile (parola −0.43, posizione +0.51, nessuna significativa); unico contrasto significativo della sua tabella: **senza highlight il prefisso è dannoso** (no_prefix −2.03 pp vs plain, p_Holm 0.001)
   - DeBERTa-lg ed Ettin: nessuna componente utile
5. **NeoBERT** (222M, architettura moderna GQA+RoPE+SwiGLU) fa il valore assoluto più alto dei nuovi (76.10, a 2.4 pp da DeBERTa-lg) e conferma il trend "large": le architetture nuove non cambiano la storia.
6. **mmBERT** (330M, multilingue 100+ lingue) è il profilo più **neutro** della tabella: nessun contrasto supera 1.1 pp e nessun contrasto è significativo (p_Holm ≥ 0.16, McNemar+bootstrap); i suoi fold std sono i più bassi in assoluto (0.56–1.70). La sua migliore condizione è `pos_only` (75.22), seguita da `word_only` (75.11): anche qui i marcatori espliciti non servono.
7. **ModernBERT** (2024, 93M, RMSNorm+GQA+SwiGLU) chiude la fascia bassa dei nuovi (69.37, sopra solo NomicBERT) nonostante l'architettura più recente: l'architettura moderna da sola non basta, e il confondimento **dimensione × pretraining × tokenizer** resta il limite dell'ablation. Il suo profilo è comunque "large trend" (marcatori neutrali/dannosi, `pos_only` la migliore condizione).

## Riserve

- I valori 🆕 sono medie su fold/OOF sul protocollo identico al notebook; i 2 modelli del notebook sono ripresi dai CSV salvati. A run completato, `--stats-only` ricalcolerà **tutti e 7 i backbone in un'unica passata** e rigenererà CSV/LaTeX definitivi.
- Le **statistiche di significatività** (McNemar esatto + correzione di Holm + CI bootstrap 95% sui contrasti, Tabella 5 del paper) sono definitive in `cv_mcnemar_bootstrap.csv` per tutti e 7 i backbone. Sintesi dei contrasti significativi per backbone: NomicBERT (6/7), UltraBERT (5/7), EuroBERT (3), NeoBERT (3), ModernBERT (1: no_prefix vs plain, p_Holm 0.001), Ettin (1: full vs no_highlight, p_Holm 0.0001), mmBERT (nessuno, p_Holm ≥ 0.16).
- Gli std sui fold di NomicBERT sono alti (3–7 pp): i suoi contrasti andranno letti con le statistiche formalizzate.
