# parallel-warehouse-order-fulfillment

## Benchmark

`mai.py` compares a serial loop, `multiprocessing.Pool.map`, and a four-thread
pipeline using bounded `queue.Queue` instances. It uses the standard library;
no third-party packages are required.

Run the full experiment (default `N = 51,000,000`):

```bash
python mai.py
```

For a smaller validation run:

```bash
python mai.py --size 40000 --repeats 1 --workers 4
```

Optional arguments are `--seed`, `--repeats`, `--workers`,
`--pipeline-chunk-size`, and `--output-dir`. The full run tests Pool worker
counts 1, 2, 4, and the selected worker count (by default, all CPUs available
to the process), with `Pool.map` chunksizes 1 and `N/(4 * CPU count)`. The
pipeline uses queue capacities 16 and 1024.

The script also benchmarks sizes `N/4`, `N/2`, and `N`. Input generation and
correctness checks are outside the timed operation. Every output is checked
against the serial result by comparing BLAKE2b digests of the complete
double-precision arrays. Results are written to `benchmark_results/`:

- `benchmark_report.md` — timing tables, throughput, speedup, efficiency, and
  interpretation;
- `benchmark_results.csv` — all measured configurations;
- `benchmark_scaling.svg` — execution-time and speedup plots across sizes.

## Fraud Detection Model

Install the dependencies and train the model offline:

```bash
pip install -r requirements.txt
python -m app.ml.train
```

Training reads `data/training.csv` and uses `is_fraud` as the target. The CSV
must contain `amount`, `currency`, `country`, `city`,
`transactions_last_10_min`, `is_new_device`, and `is_fraud`; the fraud target
must use `1` for fraud and `0` for legitimate transactions. `user_id` may be
present in the CSV but is not used as a model feature. Categorical columns are
encoded in the training pipeline, with infrequent values grouped to limit
high-cardinality expansion. The fitted pipeline is saved to `app/ml/model.pkl`.
The training script reports holdout accuracy, precision, recall, F1-score, and
ROC-AUC.

`app.ml.predict.predict_transaction(data)` loads the saved model on first use
and returns a fraud `risk_score` and `is_fraud` decision. Pass the model's
feature columns as keys in `data`; optional extra transaction fields are
ignored. `POST /transaction/add/` accepts `user_id`, `amount`, `currency`,
`country`, `city`, `transactions_last_10_min`, and `is_new_device`, as well as
the trained PaySim model fields `step`, `type`, `oldbalanceOrg`,
`newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, and `isFlaggedFraud`.
Those PaySim fields are required for inference and must be supplied from the
transaction source; they cannot be derived from the other API fields. The
response includes the transaction ID, risk score, fraud decision, and status.