# Methodology notes

These notes describe **the implemented protocol**, including where I would improve it. They are not claims that the problems have already been corrected.

## Task and data

WiC is a binary classification task over pairs of sentences containing a target word. The original notebook downloads `WiC.zip` from the public SuperGLUE distribution. It loads 5,428 training examples and 638 validation examples, combined into a labeled pool of **6,066** for the later CV stage. The hidden test labels are not used in the reported cross-validation.

## Representations

The main 2×2 comparison turns on/off two input features: (1) `<TW>...</TW>` markers around the target word, and (2) a target-word / POS text prefix. `word_only` and `pos_only` are additional attempts to separate those prefix components. Both backbones are trained under the same set of conditions, with fixed hyperparameters and model-specific tokenization.

## Evaluation

The notebook first evaluates the main configurations on the fixed validation split using three random seeds. It then creates five stratified folds from the combined labeled pool, reuses the same fold assignments across conditions, collects OOF predictions, reports accuracy and macro-F1, and compares paired predictions with exact McNemar tests. The notebook applies Holm adjustment to the implemented contrast family and reports bootstrap CIs for accuracy differences.

The outputs in the repository belong to those historical executions; the underlying Drive OOF JSON/CSV files are **not included**. Values in `results/` were extracted from saved stdout and rounded exactly as displayed.

## Important caveats

**1. Checkpoint selection on the evaluation fold.** `build_training_args()` sets `eval_strategy='epoch'` and `load_best_model_at_end=True`; the CV `WeightedTrainer` receives the outer fold as `eval_dataset`. The best checkpoint is therefore selected using that same fold. I would split each training fold again to select the checkpoint, or fix the number of epochs beforehand. Otherwise reported accuracy can be optimistic.

**2. Missing POS annotations.** `load_wic_jsonl()` uses `d.get('pos', 'N')`. Both training and validation outputs show only `N`, meaning the `pos_only` input is a constant `(noun): ...` prefix. The experiment is not evidence of the effect of genuinely varying POS labels. I would either source valid POS annotations or make the constant-prefix nature explicit in the experiment naming.

**3. Hyperparameter provenance.** The notebook states that the selected hyperparameters were tuned on DeBERTa-large and reused for the other backbone. This makes the within-backbone input-format contrasts interpretable, but absolute model-to-model comparisons are not optimized symmetrically.

**4. Data and artifact availability.** Colab/Drive are integral to the workflow. To reproduce the reported numbers, the environment, dependency versions, data version, checkpoint-selection protocol and saved intermediate results need to be pinned and rerun. The current `requirements.txt` is a dependency overview, not a validated lockfile.

**5. Plotting.** Some late plotting cells contain manually transcribed numerical values. Those figures are presentation snapshots; in future I would generate them only from the exported analysis tables.

## What I would do next

Use an inner validation split for checkpoint selection, audit the POS metadata, save dependency versions with each run, export machine-readable prediction files where distribution is permitted, and construct every figure from the same saved results table. This would turn the current exploratory notebook into a more reproducible experimental pipeline.