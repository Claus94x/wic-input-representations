#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiC 2x2 ablation — k-fold CV su NUOVI backbone (estensione del notebook)
========================================================================
Riproduce ESATTAMENTE la sezione 8 (CV-1..CV-8) di
`notebooks/wic_ablation_study.ipynb`, cambiando solo i modelli:

  * stesso dataset: WiC (SuperGLUE), pool train+dev = 6066 esempi etichettati
  * stesso split: StratifiedKFold(5, shuffle=True, random_state=CV_SEED=1234)
  * stesse 6 condizioni: full, no_highlight, no_prefix, plain, word_only, pos_only
  * stessi iperparametri Optuna del notebook (tunati su DeBERTa-large)
  * stesso seed per fold: set_seed(CV_SEED + fold)

Progetto per Apple Silicon (MPS, M2 Mac Pro 64GB): il protocollo non cambia
(FP32, batch effettivo 16, 4 epoche), ma il batch 16 viene fatto in un solo
passo MPS invece di 4 micro-batch (stessi passi ottimizzatore, meno overhead).

Resume: le predizioni OOF vengono salvate in
  results/extra_models/cv/cv_oof_preds.json
dopo OGNI fold. Se il run si interrompe, rilancia lo script: riparte da dove
si era fermato.

Uso:
  python3 train_cv_new_models.py              # addestra i run mancanti + statistiche
  python3 train_cv_new_models.py --stats-only # solo tabelle dai risultati salvati

Richiede transformers >= 4.51, < 5 (>=4.51 per i ModernBert nativi; <5 per i
modelli custom-code, incompatibili con le API rimosse in v5).
Run totali: 7 modelli x 6 condizioni x 5 fold = 210 run.
"""
import argparse, gc, json, os, shutil, sys, time, urllib.request, warnings, zipfile
from pathlib import Path

# Apple Silicon: allenta il timeout CDM del driver AGX per ridurre i crash
# "MPS command buffer execution failed" nei run lunghi (watchdog Metal).
# Va impostato prima di importare torch. Da solo non sempre basta: il
# watchdog scatta quando il display e' acceso (WindowServer compone sulla
# stessa GPU) — per run lunghi e' consigliabile display spento.
os.environ.setdefault("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1")

# Stub di xformers (vendor/) per i modelli custom-code su macOS: xformers
# non ha build per Apple Silicon. Aggiunto a FINE di sys.path: se xformers
# vero e' installato (es. Linux+GPU) quello vince.
sys.path.append(str(Path(__file__).resolve().parent / "vendor"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import evaluate
from datasets import Dataset as HFDataset
from scipy.stats import binomtest
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Config — identica al notebook (sez. 4 + CV-1/CV-2); cambiano SOLO i modelli
# ---------------------------------------------------------------------------
BACKBONES = [
    "nomic-ai/nomic-bert-2048",    # NomicBERT
    "dmis-lab/UltraBERT-150M",     # UltraBERT
    "jhu-clsp/ettin-encoder-150m", # Ettin
    "EuroBERT/EuroBERT-210m",      # EuroBERT
    "chandar-lab/NeoBERT",         # NeoBERT
    "jhu-clsp/mmBERT-base",        # mmBERT
    "answerdotai/ModernBERT-base", # ModernBERT
]
# NomicBERT, UltraBERT, EuroBERT, NeoBERT portano il proprio codice di
# modeling: serve trust_remote_code=True (inoffensivo per i modelli standard).
TRUST_REMOTE_CODE = True

MAX_LEN   = 128
USE_FP16  = False     # FP32 come nel notebook (e obbligatorio su MPS)
EPOCHS    = 4
K_FOLDS   = 5
CV_SEED   = 1234      # seed dello split (fisso) E seed di training per fold
INCLUDE_PREFIX_DECOMP = True

# Iperparametri Optuna del notebook (tunati su DeBERTa-large):
BEST_PARAMS = {
    "learning_rate": 8.655684637181687e-06,
    "weight_decay": 0.0384178539730063,
    "hidden_dropout": 0.09350951469683673,
    "attention_dropout": 0.06661222055295538,
    "warmup_ratio": 0.07598012902842396,
    "label_smoothing": 0.020917269578948443,
}

# 2x2 fattoriale + decomposizione del prefisso (stesse condizioni del CV notebook)
CV_CONDITIONS = [
    {"name": "full",         "use_highlight": True,  "prefix_mode": "full"},
    {"name": "no_highlight", "use_highlight": False, "prefix_mode": "full"},
    {"name": "no_prefix",    "use_highlight": True,  "prefix_mode": "none"},
    {"name": "plain",        "use_highlight": False, "prefix_mode": "none"},
]
if INCLUDE_PREFIX_DECOMP:
    CV_CONDITIONS += [
        {"name": "word_only", "use_highlight": False, "prefix_mode": "word"},
        {"name": "pos_only",  "use_highlight": False, "prefix_mode": "pos"},
    ]

# ---------------------------------------------------------------------------
# 2. Device (MPS su Apple Silicon) e cartelle locali (niente Google Drive)
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def empty_cache():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()

def fold_seed(fold):
    return CV_SEED + fold    # stessa regola del notebook

ROOT          = Path(__file__).resolve().parent
DATA_DIR      = ROOT / "wic_data"
RESULTS_DIR   = ROOT / "results" / "extra_models"
CV_DIR        = RESULTS_DIR / "cv"
CV_OOF_PATH   = CV_DIR / "cv_oof_preds.json"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
for d in (CV_DIR, CHECKPOINT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 3. Dataset: download + loader IDENTICI al notebook (cell. 12)
# ---------------------------------------------------------------------------
WIC_URL = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WiC.zip"

def load_wic_jsonl(fp):
    rows = []
    with open(fp, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "idx": d.get("idx", -1), "word": d["word"], "pos": d.get("pos", "N"),
                "sentence1": d["sentence1"], "sentence2": d["sentence2"],
                "start1": d["start1"], "end1": d["end1"],
                "start2": d["start2"], "end2": d["end2"],
                "label": int(d["label"]) if d.get("label") is not None else None,
            })
    return pd.DataFrame(rows)

def load_wic():
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "WiC.zip"
    if not zip_path.exists():
        print("Download dataset..."); urllib.request.urlretrieve(WIC_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    base = DATA_DIR / "WiC"
    train_df = load_wic_jsonl(base / "train.jsonl")
    dev_df   = load_wic_jsonl(base / "val.jsonl")
    print(f"Train {len(train_df)} | Dev {len(dev_df)}")
    return train_df, dev_df

# ---------------------------------------------------------------------------
# 4. Featurization: IDENTICA al notebook (cell. 16 + CV-2)
# ---------------------------------------------------------------------------
HIGHLIGHT_TOKENS = ["<TW>", "</TW>"]
POS_MAP = {"N": "noun", "V": "verb", "J": "adjective", "R": "adverb"}

def highlight_word(sentence, start, end):
    try:
        s, e = int(start), int(end)
        if not (0 <= s < e <= len(sentence)):
            return sentence
        return sentence[:s] + "<TW>" + sentence[s:e] + "</TW>" + sentence[e:]
    except Exception:
        return sentence

def add_features_cv(df, use_highlight, prefix_mode):
    df = df.copy()
    def make_sent(row, n):
        s = row[f"sentence{n}"]
        if use_highlight:
            s = highlight_word(s, row[f"start{n}"], row[f"end{n}"])
        if prefix_mode == "full":
            s = f"{row['word']} ({POS_MAP.get(row['pos'], 'word')}): " + s
        elif prefix_mode == "word":
            s = f"{row['word']}: " + s
        elif prefix_mode == "pos":
            s = f"({POS_MAP.get(row['pos'], 'word')}): " + s
        return s
    df["s1_final"] = df.apply(lambda r: make_sent(r, 1), axis=1)
    df["s2_final"] = df.apply(lambda r: make_sent(r, 2), axis=1)
    return df

def build_pool_split(train_df, dev_df):
    """CV-1: pool train+dev + split stratificato UNICO (stesso seed del notebook)."""
    pool_df = pd.concat([train_df, dev_df], ignore_index=True).reset_index(drop=True)
    GOLD = pool_df["label"].values.astype(int)
    N_POOL = len(pool_df)
    assert pool_df["label"].notna().all(), "Label mancanti nel pool train+dev."
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=CV_SEED)
    FOLD_OF = np.empty(N_POOL, dtype=int)
    for fold_id, (_tr_idx, te_idx) in enumerate(skf.split(np.zeros(N_POOL), GOLD)):
        FOLD_OF[te_idx] = fold_id
    print(f"Pool train+dev: {N_POOL} esempi | {K_FOLDS} fold | bilanciamento: "
          f"{np.bincount(GOLD)}")
    return pool_df, GOLD, FOLD_OF

# ---------------------------------------------------------------------------
# 5. Helper di training: IDENTICI al notebook (cell. 28), adatti a MPS
# ---------------------------------------------------------------------------
acc_metric = evaluate.load("accuracy")
f1_metric  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]}

class WeightedTrainer(Trainer):
    """Trainer con cross-entropy pesata + label smoothing (cell. 28 notebook)."""
    def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=w, label_smoothing=self.label_smoothing)(
            logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def build_training_args(out_dir, seed, n_train_samples):
    # Batch EFFETTIVO 16, identico al notebook (4 x accum 4): su M2 64GB lo
    # facciamo in un solo batch MPS => stessi passi ottimizzatore, meno overhead.
    # transformers v5 ha rimosso warmup_ratio: lo convertiamo in warmup_steps
    # con la stessa formula (ratio * n_passi_totali) => protocollo invariato.
    steps_per_epoch = -(-n_train_samples // 16)   # ceil
    warmup_steps = int(BEST_PARAMS["warmup_ratio"] * steps_per_epoch * EPOCHS)
    return TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=EPOCHS,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=BEST_PARAMS["learning_rate"],
        weight_decay=BEST_PARAMS["weight_decay"],
        warmup_steps=warmup_steps, lr_scheduler_type="cosine",
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="accuracy", greater_is_better=True,
        fp16=USE_FP16, bf16=False, gradient_checkpointing=False,
        logging_steps=50, report_to="none", save_total_limit=1, seed=seed,
        dataloader_num_workers=2, disable_tqdm=False,
    )

def apply_dropout_config(cfg):
    """Dropout del notebook applicato al nome dell'attributo usato dal modello
    (BERT-style: hidden_dropout_prob / ModernBert: hidden_dropout)."""
    for attr, val in (
        ("hidden_dropout_prob", BEST_PARAMS["hidden_dropout"]),
        ("hidden_dropout",      BEST_PARAMS["hidden_dropout"]),
        ("attention_probs_dropout_prob", BEST_PARAMS["attention_dropout"]),
        ("attention_dropout",   BEST_PARAMS["attention_dropout"]),
    ):
        if hasattr(cfg, attr):
            setattr(cfg, attr, val)

def resize_embeddings(model, new_len):
    """Ridimensiona le word embeddings per i token <TW></TW> (no-op se assenti).
    Fallback per modelli custom senza get_input_embeddings (limitazione di v5):
    le righe nuove sono inizializzate come N(media, std) dei pesi vecchi,
    comportamento equivalente al resize di v4 con mean_resizing=False."""
    try:
        model.resize_token_embeddings(new_len, mean_resizing=False)
        return
    except (NotImplementedError, AttributeError):
        # modelli custom senza get_input_embeddings (limitazione di v5 / codice 2024)
        pass
    embs = [m for m in model.modules() if isinstance(m, nn.Embedding)]
    emb = max(embs, key=lambda m: m.num_embeddings)   # le word embeddings sono la piu' grande
    if new_len <= emb.num_embeddings:
        return
    w = emb.weight
    with torch.no_grad():
        new_w = w.new_empty((new_len, w.shape[1]))
        new_w[: w.shape[0]].copy_(w)
        tail = new_w[w.shape[0]:]
        if tail.numel():
            tail.normal_(mean=w.mean().item(), std=w.std().item())
    emb.weight = nn.Parameter(new_w)

def make_model(model_name, tokenizer):
    cfg = AutoConfig.from_pretrained(model_name, num_labels=2,
                                     trust_remote_code=TRUST_REMOTE_CODE)
    apply_dropout_config(cfg)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=cfg, ignore_mismatched_sizes=True,
            trust_remote_code=TRUST_REMOTE_CODE)
    except RuntimeError as e:
        if "Missing key(s) in state_dict" not in str(e):
            raise
        # Checkpoint di pretraining senza pooler/classifier (modelli custom
        # come NomicBERT): il codice custom carica in strict mode. Ricarico
        # con strict=False: i pesi mancanti restano inizializzati a caso,
        # esattamente come nel notebook (v4, ignore_mismatched_sizes=True).
        print(f"[warn] {model_name}: checkpoint pretraining, ricarica con strict=False")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=cfg, ignore_mismatched_sizes=True,
            trust_remote_code=TRUST_REMOTE_CODE, strict=False)
    # stesso comportamento di v4 (mean_resizing=False) usato dal notebook
    resize_embeddings(model, len(tokenizer))
    return model

def load_tokenizer(model_name):
    kwargs = {}
    try:
        import json as _json
        from huggingface_hub import hf_hub_download
        tc = _json.load(open(hf_hub_download(model_name, "tokenizer_config.json")))
        if isinstance(tc.get("extra_special_tokens"), list):
            # config tipo UltraBERT: extra_special_tokens come LISTA, ma questa
            # versione di transformers si aspetta un dizionario {nome: token}.
            # Registriamo ogni token col suo nome generato: stesso effetto.
            kwargs["extra_special_tokens"] = {f"ext_special_{i}": t
                                              for i, t in enumerate(tc["extra_special_tokens"])}
    except Exception as e:
        print(f"[warn] {model_name}: tokenizer_config non leggibile, carico standard: {e}")
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=TRUST_REMOTE_CODE, **kwargs)

def build_cv_data(model_name, cond, tr_df, ev_df):
    """CV-3: tokenizza uno split arbitrario (mirror di build_condition_data)."""
    tok = load_tokenizer(model_name)
    if cond["use_highlight"]:
        tok.add_special_tokens({"additional_special_tokens": HIGHLIGHT_TOKENS})
    tr = add_features_cv(tr_df, cond["use_highlight"], cond["prefix_mode"])
    ev = add_features_cv(ev_df, cond["use_highlight"], cond["prefix_mode"])
    def tok_fn(ex):
        out = tok(ex["s1_final"], ex["s2_final"], max_length=MAX_LEN, truncation=True, padding=False)
        out["labels"] = ex["label"]; return out
    cols = ["s1_final", "s2_final", "label"]
    tr_d = HFDataset.from_pandas(tr[cols].reset_index(drop=True)).map(tok_fn, remove_columns=cols)
    ev_d = HFDataset.from_pandas(ev[cols].reset_index(drop=True)).map(tok_fn, remove_columns=cols)
    return tok, tr_d, ev_d

# ---------------------------------------------------------------------------
# 6. Loop k-fold RESUMABILE (CV-4 notebook): salva OOF su disco dopo ogni fold
# ---------------------------------------------------------------------------
def load_oof():
    if CV_OOF_PATH.exists():
        return json.load(open(CV_OOF_PATH))
    return {}   # chiave "backbone||cond||fold" -> {"idx": [...], "pred": [...]}

def save_oof(d):
    with open(CV_OOF_PATH, "w") as f:
        json.dump(d, f)

def run_cv(pool_df, GOLD, FOLD_OF, folds=None):
    oof = load_oof()
    total = len(BACKBONES) * len(CV_CONDITIONS) * K_FOLDS
    print(f"Run gia' completati: {len(oof)}/{total}")
    for backbone in BACKBONES:
        for cond in CV_CONDITIONS:
            cname = cond["name"]
            for fold in (range(K_FOLDS) if folds is None else folds):
                key = f"{backbone}||{cname}||fold{fold}"
                if key in oof:
                    print(f"[skip] {key}")
                    continue
                te_mask = (FOLD_OF == fold)
                tr_df_f = pool_df[~te_mask].reset_index(drop=True)
                ev_df_f = pool_df[te_mask].reset_index(drop=True)
                ev_pool_idx = np.where(te_mask)[0].tolist()   # indici nel pool (per appaiare)

                print(f"\n=== {backbone} | {cname} | fold {fold} "
                      f"(train {len(tr_df_f)}, eval {len(ev_df_f)}) ===")
                set_seed(fold_seed(fold)); gc.collect(); empty_cache()

                tok, tr_d, ev_d = build_cv_data(backbone, cond, tr_df_f, ev_df_f)

                # class weights ricalcolati sul TRAIN del fold (WiC ~bilanciato => ~neutri)
                cw = compute_class_weight("balanced", classes=np.array([0, 1]),
                                          y=tr_df_f["label"].values)
                class_weights = torch.tensor(cw, dtype=torch.float)

                out_dir = CHECKPOINT_DIR / f"cv_{backbone.split('/')[-1]}_{cname}_fold{fold}"
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                out_dir.mkdir(parents=True)

                model = make_model(backbone, tok)
                trainer = WeightedTrainer(
                    model=model, args=build_training_args(out_dir, fold_seed(fold), len(tr_df_f)),
                    train_dataset=tr_d, eval_dataset=ev_d,
                    data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
                    class_weights=class_weights, label_smoothing=BEST_PARAMS["label_smoothing"])
                t0 = time.time(); trainer.train(); mins = (time.time() - t0) / 60
                preds = np.argmax(trainer.predict(ev_d).predictions, axis=-1).astype(int).tolist()

                oof[key] = {"idx": ev_pool_idx, "pred": preds}
                save_oof(oof)   # salvataggio dopo OGNI fold (resume)
                acc = float(np.mean(np.array(preds) == GOLD[ev_pool_idx]))
                print(f"  >>> fold acc={acc*100:.2f}%  {mins:.1f} min  [OOF salvato]")

                del model, trainer; gc.collect(); empty_cache()
                shutil.rmtree(out_dir, ignore_errors=True)
    print("\nCV COMPLETATA (o ripresa: rilancia se mancano run).")

# ---------------------------------------------------------------------------
# 7. Statistiche: CV-5/CV-6/CV-7/CV-8 del notebook, identiche
# ---------------------------------------------------------------------------
def mcnemar_exact(pa, pb, g):
    a, b = np.array(pa), np.array(pb)
    n01 = int(np.sum((a == g) & (b != g)))
    n10 = int(np.sum((a != g) & (b == g)))
    n = n01 + n10
    p = 1.0 if n == 0 else binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue
    return n01, n10, p

def holm(pvals):
    m = len(pvals); order = sorted(range(m), key=lambda i: pvals[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj

def bootstrap_delta_ci(va, vb, g, n_boot=10000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    base = (va == g).astype(float) - (vb == g).astype(float)
    n = len(g)
    deltas = np.array([base[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(base.mean() * 100), float(lo * 100), float(hi * 100)

CONTRASTS = [
    ("highlight_effect_with_prefix",    "full",         "no_highlight"),
    ("highlight_effect_without_prefix", "no_prefix",    "plain"),
    ("prefix_effect_with_highlight",    "full",         "no_prefix"),
    ("prefix_effect_without_highlight", "no_highlight", "plain"),
]
if INCLUDE_PREFIX_DECOMP:
    CONTRASTS += [
        ("word_component",  "word_only", "plain"),
        ("pos_component",   "pos_only",  "plain"),
        ("word_vs_pos",     "word_only", "pos_only"),
    ]

def report(pool_df, GOLD):
    N_POOL = len(pool_df)
    oof = load_oof()

    def assemble_oof(backbone, cname):        # CV-5
        full_pred = np.full(N_POOL, -1, dtype=int)
        for fold in range(K_FOLDS):
            key = f"{backbone}||{cname}||fold{fold}"
            if key not in oof:
                return None
            d = oof[key]
            full_pred[np.array(d["idx"])] = np.array(d["pred"])
        return None if np.any(full_pred < 0) else full_pred

    # Include anche i modelli gia' presenti nel file OOF ma non in BACKBONES
    # (es. run eseguito con un altro ambiente/venv, vedi --backbone).
    known = list(dict.fromkeys(list(BACKBONES) + [k.split("||")[0] for k in oof]))
    backbones_seen = [b for b in known
                      if all(assemble_oof(b, c["name"]) is not None for c in CV_CONDITIONS)]
    if not backbones_seen:
        print("Nessun backbone con CV completa: nulla da analizzare.")
        return
    OOF_PRED = {(b, c["name"]): assemble_oof(b, c["name"])
                for b in backbones_seen for c in CV_CONDITIONS}
    print("Backbone con CV completa:", backbones_seen)

    # CV-6: tabelle accuratezza/F1 per condizione
    summary_rows = []
    for backbone in backbones_seen:
        perfold = {c["name"]: {"acc": [], "f1": []} for c in CV_CONDITIONS}
        for cond in CV_CONDITIONS:
            for fold in range(K_FOLDS):
                d = oof[f"{backbone}||{cond['name']}||fold{fold}"]
                idx = np.array(d["idx"]); pr = np.array(d["pred"]); g = GOLD[idx]
                perfold[cond["name"]]["acc"].append(np.mean(pr == g))
                perfold[cond["name"]]["f1"].append(f1_score(g, pr, average="macro"))
        full_acc_pooled = np.mean(OOF_PRED[(backbone, "full")] == GOLD)
        print("=" * 74); print(f"Backbone: {backbone}  (CV {K_FOLDS}-fold, N={N_POOL})")
        print(f"{'Condizione':<14}{'Acc OOF':<10}{'Acc media±std':<20}{'F1 media±std':<20}{'Δ vs full':<10}")
        for cond in CV_CONDITIONS:
            cn = cond["name"]
            pooled_acc = np.mean(OOF_PRED[(backbone, cn)] == GOLD)
            a = np.array(perfold[cn]["acc"]); f = np.array(perfold[cn]["f1"])
            delta = (pooled_acc - full_acc_pooled) * 100
            print(f"{cn:<14}{pooled_acc*100:>6.2f}    {a.mean()*100:>5.2f} ± {a.std(ddof=1)*100:4.2f}     "
                  f"{f.mean()*100:>5.2f} ± {f.std(ddof=1)*100:4.2f}     {delta:+.2f} pp")
            summary_rows.append({
                "backbone": backbone, "condition": cn, "n_pool": int(N_POOL),
                "acc_oof": float(pooled_acc), "acc_fold_mean": float(a.mean()),
                "acc_fold_std": float(a.std(ddof=1)), "f1_fold_mean": float(f.mean()),
                "f1_fold_std": float(f.std(ddof=1)), "delta_vs_full_pp": float(delta)})
    pd.DataFrame(summary_rows).to_csv(CV_DIR / "cv_ablation_summary.csv", index=False)

    # CV-7: McNemar + Holm + CI bootstrap sugli OOF
    mc_rows = []
    for backbone in backbones_seen:
        print("=" * 92); print(f"Backbone: {backbone}  (CV {K_FOLDS}-fold, McNemar su N={N_POOL})")
        staged, pvals = [], []
        for name, a, b in CONTRASTS:
            va, vb = OOF_PRED[(backbone, a)], OOF_PRED[(backbone, b)]
            n01, n10, p = mcnemar_exact(va, vb, GOLD)
            d, lo, hi = bootstrap_delta_ci(va, vb, GOLD)
            staged.append((name, a, b, n01, n10, d, lo, hi)); pvals.append(p)
        padj = holm(pvals)
        for (name, a, b, n01, n10, d, lo, hi), p, pa in zip(staged, pvals, padj):
            sig = "SIGNIFICATIVO" if pa < 0.05 else "ns"
            print(f"  {a:>12} vs {b:<12}  Δ={d:+5.2f}pp  CI[{lo:+.2f},{hi:+.2f}]  "
                  f"p={p:.4f}  Holm={pa:.4f}  {sig}")
            mc_rows.append({"backbone": backbone, "contrast": name, "condition_A": a,
                            "condition_B": b, "A_only": n01, "B_only": n10,
                            "delta_acc_pp": d, "ci95_lo_pp": lo, "ci95_hi_pp": hi,
                            "p_value": p, "p_holm": pa, "significant_holm_005": bool(pa < 0.05)})
    pd.DataFrame(mc_rows).to_csv(CV_DIR / "cv_mcnemar_bootstrap.csv", index=False)

    # CV-8: tabelle LaTeX pronte per il paper
    fmt = lambda x, d=2: f"{x:.{d}f}"
    print("\n% ---- Tabella 4 (CV) ----")
    print(r"\begin{table}[t]\centering\caption{Risultati in validazione incrociata "
          f"{K_FOLDS}-fold su train+dev (N={N_POOL}). Media e deviazione standard sui fold.}}")
    print(r"\label{tab:cv_results}\begin{tabular}{llccc}\toprule")
    print(r"Backbone & Condizione & Acc.\ OOF (\%) & Acc.\ media$\pm$std (\%) & $\Delta$ vs.\ full (pp)\\\midrule")
    for r in summary_rows:
        print(f"{r['backbone'].split('/')[-1]} & {r['condition']} & {fmt(r['acc_oof']*100)} & "
              f"{fmt(r['acc_fold_mean']*100)} $\\pm$ {fmt(r['acc_fold_std']*100)} & "
              f"{r['delta_vs_full_pp']:+.2f}\\\\")
    print(r"\bottomrule\end{tabular}\end{table}")
    print("\n% ---- Tabella 5 (CV) ----")
    print(r"\begin{table}[t]\centering\caption{Contrasti fattoriali in CV "
          f"{K_FOLDS}-fold: McNemar con correzione di Holm e CI bootstrap 95\\% (N={N_POOL}).}}")
    print(r"\label{tab:cv_stats}\begin{tabular}{llcccc}\toprule")
    print(r"Backbone & Contrasto & $\Delta$ Acc.\ (pp) & CI 95\% (pp) & $p_{\text{Holm}}$ & Sig.\\\midrule")
    for r in mc_rows:
        sig = r"$\checkmark$" if r["significant_holm_005"] else "--"
        print(f"{r['backbone'].split('/')[-1]} & {r['condition_A']} vs.\\ {r['condition_B']} & "
              f"{r['delta_acc_pp']:+.2f} & [{r['ci95_lo_pp']:+.2f}, {r['ci95_hi_pp']:+.2f}] & "
              f"{fmt(r['p_holm'], 4)} & {sig}\\\\")
    print(r"\bottomrule\end{tabular}\end{table}")
    print(f"\nSalvato: {CV_DIR / 'cv_ablation_summary.csv'}")
    print(f"Salvato: {CV_DIR / 'cv_mcnemar_bootstrap.csv'}")

# ---------------------------------------------------------------------------
# 8. main
# ---------------------------------------------------------------------------
def main():
    global BACKBONES
    ap = argparse.ArgumentParser(description="WiC k-fold CV ablation su nuovi backbone")
    ap.add_argument("--stats-only", action="store_true",
                    help="salta il training: ricalcola le tabelle dagli OOF salvati")
    ap.add_argument("--folds", type=int, nargs="+", default=None, metavar="N",
                    help="addestra solo i fold indicati (es. --folds 0: test/benchmark)")
    ap.add_argument("--backbone", nargs="+", default=None, metavar="HF_ID",
                    help="addestra solo questi modelli (es. --backbone dmis-lab/UltraBERT-150M)")
    args = ap.parse_args()
    if args.folds is not None:
        bad = [f for f in args.folds if not 0 <= f < K_FOLDS]
        if bad:
            ap.error(f"--folds fuori range [0, {K_FOLDS-1}]: {bad}")
    if args.backbone is not None:
        bad = [b for b in args.backbone if b not in BACKBONES]
        if bad:
            ap.error(f"--backbone non in BACKBONES: {bad}\nOpzioni: {BACKBONES}")
        BACKBONES = args.backbone

    print(f"Device: {DEVICE} | Torch: {torch.__version__}")
    train_df, dev_df = load_wic()
    pool_df, GOLD, FOLD_OF = build_pool_split(train_df, dev_df)

    if args.stats_only:
        report(pool_df, GOLD)
        return

    t_start = time.time()
    run_cv(pool_df, GOLD, FOLD_OF, folds=args.folds)
    print(f"\nTempo totale training: {(time.time()-t_start)/3600:.2f} h")
    if args.folds is not None:
        print(f"AVVISO: eseguiti solo i fold {args.folds}; le tabelle richiedono tutti i {K_FOLDS}.")
    report(pool_df, GOLD)

if __name__ == "__main__":
    main()
