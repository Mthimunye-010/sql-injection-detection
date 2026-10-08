# AGNCS Django prototype

This local Django app runs the notebook's trained **M7_AGNCS_full_seed0** model. Paste Python code to receive the model's vulnerability probability, token-attention aggregation by source line, and top five nearest examples from a bundled 192-function reference index using the trained search embedding.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python manage.py runserver
```

Open http://127.0.0.1:8000. Model weights load on the first analysis; first search also computes embeddings for the small reference index.

## Scope

This is inference, not a retraining interface. Its checkpoint and vocabulary are from the supplied notebook project. Attention is a visualization of pooling weights, not a causal explanation. The model was trained on a synthetic dataset, and the notebook reports mixed out-of-distribution detection performance. Review findings manually; this prototype is not a substitute for security review or a dedicated static analyzer.
