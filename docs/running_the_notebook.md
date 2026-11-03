# Running the notebook

The notebook was developed for **Google Colab**. A local execution is not plug-and-play because the original setup imports `google.colab.drive` and writes paths under `/content/drive/MyDrive/wic_paper`.

1. Upload/open `notebooks/wic_ablation_study.ipynb` in Colab.
2. Choose a GPU runtime and run the setup and dependency-installation cells. The recorded session used a Tesla T4, but this does **not** guarantee that every new run will fit on the same hardware.
3. Allow the Drive mount and inspect `PERSIST_DIR` in the configuration cell. Outputs are saved under `MyDrive/wic_paper/`.
4. The dataset is downloaded directly from `https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WiC.zip`. The notebook expects `wic_data/WiC/{train,val,test}.jsonl`.
5. Run cells in order. The original notebook has a three-seed fixed-validation experiment and a separate five-fold CV experiment. Some plotting cells near the end use fixed values rather than dynamically loading results.
6. Before running training again, back up or inspect existing Drive results: the code has resume logic and may skip runs already recorded in Drive.

**Important:** the saved notebook opens with previous outputs already displayed. The main CV cell in that saved session printed `60/60` completed runs and skipped training. The repository does not contain model checkpoints or OOF JSON files. Starting on a new Drive will require completing the runs to regenerate those artifacts.

`requirements.txt` is a list of libraries, not a lockfile for an environment that has been freshly tested. For rigorous reproducibility, capture exact versions and rerun the full protocol with corrected checkpoint selection.
