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

`app.ml.predict.predict_transaction(data)` loads
`app/ml/cuda_train_fast.pkl` on first use and returns the raw model probability
(`model_risk_score`), trained fraud threshold (`fraud_threshold`), contextual
risk (`contextual_risk_score`), combined review risk (`risk_score`), and the
model-only `is_fraud` decision. The version-3 CUDA artifact uses only the PaySim fields
shared with the transaction API: `step`, `amount`, `oldbalanceOrg`,
`oldbalanceDest`, `isFlaggedFraud`, and categorical `type`. Other API fields
such as currency, country, city, recent transaction count, and device status
are not learned model inputs because the current PaySim training dataset does
not contain them. Recent-transaction and device data now feed the separate,
explicit contextual risk overlay described below.
The response includes up to three model-based reasons, each identifying the
feature, its transaction value, the comparison baseline, and its impact on the
risk score. The frontend shows these under “Shubhalilikka ta’sir qilgan
xususiyatlar”. These indicate model influence, not proof of fraud or causal
explanations.
`POST /transaction/add/` accepts sender/receiver cards and transaction
context. The API obtains `oldbalanceOrg` and `oldbalanceDest` from the sender
and receiver card balances, sets the transfer type and PaySim-range hour
bucket itself, and does not accept caller-supplied fraud flags or device/
velocity values. Its response returns model probability, contextual score,
combined risk, both thresholds, and the final review status. It counts this
user's transactions in the previous ten
minutes from the database and compares a stable browser device ID (sent in
`X-Device-ID`) against hashed, per-user device records. The user agent is
stored as the device label; the browser ID itself is not stored. A new device
and recent attempts contribute a documented contextual risk overlay. The
current trained model still consumes only its PaySim training features; these
additional context signals are applied outside the model until a model is
retrained on representative device/velocity data. The `step` value is
generated from the current UTC hour and wrapped into the model's 1–744 range.
The overlay score is the
maximum of the model probability and a capped context score: 0.20 for a new
device plus 0.10 per preceding transaction in the last ten minutes (maximum
0.70). Scores of 0.50 or higher are held for review; model-positive transfers
are always held.

The recipient-card lookup endpoint returns only the cardholder's first name
and last-name initial. The dashboard displays that name after a valid card
number is entered.

### Telegram review and approval

For private admin alerts and the approve/contact buttons, configure the
following private environment variables:

```text
TELEGRAM_BOT_TOKEN=<bot token>
TELEGRAM_ADMIN_ID=<numeric admin chat/user ID>
TELEGRAM_WEBHOOK_URL=https://<public-host>/telegram/webhook
TELEGRAM_WEBHOOK_SECRET=<random secret of at least 32 characters>
```

Generate a Telegram webhook secret, for example, with
`openssl rand -hex 32`.
The admin must start the bot once in a private chat before the bot can message
them. The application registers the webhook at startup. The webhook requires
Telegram's secret-token header and accepts callbacks only from the configured
admin in the admin's private chat. A suspicious transfer remains pending and
does not reserve or debit balance. The alert includes the sender's name and
phone, card suffixes, recipient name, location, recent-attempt count, device
label, and risk reasons. The admin can retrieve the phone number from the
contact button and must independently call and verify the customer before
pressing the approval button. That one-time action atomically debits and
credits the internal account balances and records the admin decision.
Notification failures are logged and surfaced in the transfer response.
The alert separates model probability and its trained fraud threshold from
the context score and final review risk, and lists the actual feature values
passed to the model. Combined risk is the maximum of model probability and
context score; it is held for review at 50%, while the model fraud verdict
uses the threshold stored in the trained artifact.

This repository implements a local demo ledger, not settlement through a
real bank or card network. The Telegram approval must not be used to move
real funds without a bank's authenticated identity-verification process,
authorization controls, and payment-rail integration.

### User login and banking dashboard

Users register with a phone number and password; the API stores the password
as a PBKDF2-SHA256 hash and creates a default card. They log in with their
phone number and password at `POST /user/login/`, which accepts JSON and
returns a 30-minute bearer token. The protected dashboard uses
`GET /user/profile/` for profile details, `GET /user/cards/` for the logged-in
user's active card numbers, cardholder names, and balances, and
`GET /user/balance/` for the combined active-card balance. `GET /user/me/` and
`GET /user/get/profil` are profile aliases.
`GET /transaction/history/` returns recent incoming and outgoing transfers.
Transfers send the selected `sender_card_id` to `POST /transaction/add/`.
The frontend has separate `/login/` and `/register/` pages; the protected
dashboard is served at `/`.
Registration and login pages display `+998` as a fixed phone prefix and ask
the user to enter only the remaining nine digits; both submit the normalized
`+998XXXXXXXXX` number to the API.
Set a private `JWT_SECRET_KEY` in the project `.env` before starting the API;
generate a strong value with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
The token is required for transfers and transaction-detail requests, and a
transfer can only be sent from the logged-in user's own card.

### CUDA Training

The optional PyTorch trainer uses the NVIDIA GPU to train a mini-batch
fraud-detection model and reports accuracy, precision, recall, F1, ROC-AUC,
training throughput, and elapsed times. Install a CUDA-enabled PyTorch build
compatible with your NVIDIA driver, then run from the project root:

```bash
venv/bin/python -m app.ml.cuda_train
```

The model weights, preprocessing values, evaluation metrics, and timing
results are saved together in `app/ml/cuda_train_fast.pkl`. This version-3
artifact is the one loaded by the prediction API.