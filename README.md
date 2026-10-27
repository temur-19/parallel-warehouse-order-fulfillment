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
```

Training reads `data/PS_20174392719_1491204439457_log.csv` and uses
`isFraud` as the target (`1` for fraud, `0` for legitimate transactions).
Categorical columns are one-hot encoded, with infrequent values grouped to
limit high-cardinality expansion. The trainer uses MPI to divide the training
data between processes.

Because histogram gradient boosting requires dense input, the encoded
features are converted to a dense matrix during training. The trained model is
saved to `app/ml/model.pkl`. The training script reports holdout
accuracy, precision, recall, F1-score, and ROC-AUC.

Install an MPI implementation (such as Open MPI) on every participating
machine, then run from the project root:

```bash
mpiexec -n 4 venv/bin/python -m app.ml.train
```

Rank 0 reads and splits the dataset, then sends one stratified training shard
to each MPI process. Each process fits one model with native CPU threading
limited to one thread; rank 0 averages their fraud probabilities, evaluates
the ensemble, and saves it to `app/ml/model.pkl`. This artifact is separate
from the CUDA model used by the prediction API. Use the same project code and
dataset path on all ranks; only rank 0 reads the dataset. The MPI ensemble
trains on a fraction of the training rows per model, so compare its evaluation
metrics with prior results before using it in production.

`app.ml.predict.predict_transaction(data)` loads `app/ml/cuda_train.pkl` on
first use and returns a fraud `risk_score` and `is_fraud` decision. This
version-1 CUDA artifact expects the seven PaySim numeric features and the
categorical `type` feature; optional extra transaction fields are ignored.
The response includes up to three model-based reasons, each identifying the
feature, its transaction value, the comparison baseline, and its impact on the
risk score. The frontend shows these under “Shubhalilikka ta’sir qilgan
xususiyatlar”. These indicate model influence, not proof of fraud or causal
explanations.
`POST /transaction/add/` accepts `user_id`, `amount`, `currency`,
`country`, `city`, `transactions_last_10_min`, and `is_new_device`, as well as
the trained PaySim model fields `step`, `type`, `oldbalanceOrg`,
`newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, and `isFlaggedFraud`.
Those PaySim fields are required for inference and must be supplied from the
transaction source; they cannot be derived from the other API fields. The
response includes the transaction ID, risk score, fraud decision, and status.

### CUDA Training

The optional PyTorch trainer uses the NVIDIA GPU to train a mini-batch
fraud-detection model and reports accuracy, precision, recall, F1, ROC-AUC,
training throughput, and elapsed times. Install a CUDA-enabled PyTorch build
compatible with your NVIDIA driver, then run from the project root:

```bash
venv/bin/python -m app.ml.cuda_train
```

The model weights, preprocessing values, evaluation metrics, and timing
results are saved together in `app/ml/cuda_train_fast.pkl`. This CUDA artifact
is a separate version-2 artifact. The API currently loads the version-1
artifact at `app/ml/cuda_train.pkl`; it does not load `cuda_train_fast.pkl`.