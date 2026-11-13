"""Stub minimale di xformers per Apple Silicon (xformers non ha build per macOS).
Fornisce solo xformers.ops.SwiGLU — la matematica e' identica a quella di
xformers.ops.SwiGLU in packed mode (w12/w3) — e basta per i modelli
custom-code che lo importano (es. chandar-lab/NeoBERT).
Aggiunto a FINE di sys.path dallo script, cosi' se xformers vero e'
installato (Linux/GPU) quello vince sempre."""
