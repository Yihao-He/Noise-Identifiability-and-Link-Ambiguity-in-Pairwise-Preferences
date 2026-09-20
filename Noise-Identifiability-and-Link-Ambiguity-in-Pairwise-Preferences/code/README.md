# Inherited K=3,4 experiment support

The current revision's full instructions are in the root `README.md`; new experiments are in `experiments/`.

These older routines support the saved K=3,4 evidence retained in the supplement:

```text
python code/reproduce_tables.py
python code/verify_key_claims.py
python code/plot_figures.py
```

They use the root `requirements.txt` and the shipped `data/` inputs. They reanalyze or verify 3,200 previously saved binomial datasets, compute deterministic information values, and recreate the two historical figures. They do not generate new Monte Carlo datasets or call language models. The new figure/table generator is `experiments/plot_results.py`.

The inherited logistic fit domain was eta in [0,0.49] with utility coordinates in [-10,10]. These are historical numerical settings, distinct from both the new finite-sample protocol and the theorem's fixed interior b domain. The old `conservative_interval` function implements the **logistic-only** C_L; it is not the new link-robust C_H.

`original/inference.py` supplies the stable binomial-deviance and analytic derivative routine reused by the new likelihood experiments. Numeric judge-cache summaries are included, but raw third-party text and original model calls are not reproduced by these commands.
