# Word-in-Context — Studio delle rappresentazioni di input

**Python · NLP · Transformers · DeBERTa-large · RoBERTa-large**

Questo repository raccoglie il notebook che ho usato per studiare l'effetto della rappresentazione dell'input sul task Word-in-Context. Il problema è capire se la stessa parola viene usata con lo stesso significato in due frasi. Invece di modificare l'architettura, ho provato a cambiare il modo in cui le frasi vengono passate al modello.

![Risultati in cross-validation](assets/cv_accuracy.svg)

## Esperimenti

Le condizioni principali sono quattro: `full` (marker e prefisso), `no_highlight` (solo prefisso), `no_prefix` (solo marker) e `plain` (nessuno dei due). Ho considerato due backbone, DeBERTa-large e RoBERTa-large, prima con tre seed sul validation set fisso e successivamente con **5-fold cross-validation su 6.066 esempi**. Sono presenti anche le condizioni `word_only` e `pos_only`.

Non mi sono limitato all'accuracy: nel notebook ho raccolto le predizioni OOF, calcolato macro-F1, utilizzato il test di McNemar con correzione di Holm e intervalli bootstrap, e analizzato alcune differenze tra gli errori.

## Il risultato che mi interessava

| Condizione | DeBERTa-large | RoBERTa-large |
| --- | ---: | ---: |
| `full` | 78,55% | 77,37% |
| `no_highlight` | **80,81%** | **77,88%** |
| `no_prefix` | 78,52% | 76,66% |
| `plain` | 79,29% | 75,70% |

Con DeBERTa-large, eliminare i marker dalla condizione `full` porta a **+2,26 punti percentuali** di accuracy OOF. Nel confronto del notebook la differenza è significativa dopo la correzione di Holm. Non ottengo la stessa evidenza statistica per il confronto corrispondente su RoBERTa-large.

Questi numeri sono tratti dagli output salvati nel notebook, **non da una nuova esecuzione**.

## Dove guardare

- [`notebooks/wic_ablation_study.ipynb`](notebooks/wic_ablation_study.ipynb): notebook principale con codice e appunti personali sui passaggi più tecnici.
- [`archive/wic_research_notebook_with_study_notes.ipynb`](archive/wic_research_notebook_with_study_notes.ipynb): versione integrale con tutte le celle del notebook originale, comprese quelle vuote.
- [`results/`](results/): risultati trascritti dagli output salvati.
- [`docs/methodology.md`](docs/methodology.md) e [`docs/running_the_notebook.md`](docs/running_the_notebook.md): metodologia, limiti e istruzioni.

## Cosa non considero ancora risolto

Nel codice della cross-validation seleziono il checkpoint usando il fold sul quale misuro anche le prestazioni. Questa scelta può dare stime ottimistiche; per una valutazione indipendente userei una validation interna al training fold. Un altro limite è il campo POS: il loader imposta `N` come default e gli output mostrano `N` per tutti gli esempi. La condizione `pos_only` diventa quindi un prefisso costante, non un vero confronto fra categorie grammaticali.

Il notebook è pensato per **Google Colab con GPU e Google Drive**. Gli addestramenti e i checkpoint non sono inclusi nel repository; il dataset viene scaricato dal link pubblico presente nel codice. Le celle non vuote del notebook principale mantengono il **codice originale identico**, mentre la versione integrale è conservata in `archive/`.

Lavoro collegato: *Optimizing Input Representations for Word-in-Context: A Controlled Ablation Study* (AIxIA 2026), con Claudio Daloiso, Francesca Fallucchi, Manuel Gozzi e Andrea Ligi. Questo repository raccoglie notebook e appunti, non sostituisce l'articolo scientifico.

[English version](README.md)