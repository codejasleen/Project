# Warranty review public demo

This repository is a public-safe demonstration of a local warranty-claim review service. It contains only fictional identifiers, synthetic reference data and a hand-authored demonstration model.

The real trained model and assignment results are kept outside this public demonstration because they were trained on private engagement data.The production-trained model, operational datasets, predictions and engagement deliverables are intentionally excluded. The supplied engagement policy prohibits publishing customer or operational data.

This demo is suitable for showing the application structure and API workflow. Its scores are illustrative and must not be interpreted as validated fraud estimates.

## Run locally

Requirements: Python 3.11 or 3.12.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

## API

`POST /predict` accepts one fictional demo claim and returns:

- an illustrative risk score;
- a human-review recommendation;
- higher- and lower-risk signals derived from the demo model;
- warnings for unknown synthetic partner or product values; and
- demo model/version information.

Example request:

```json
{
  "claim_id": "SYNTH-CLAIM-001",
  "submitted_at": "2030-01-15 10:00",
  "partner_id": "DEMO-PARTNER-ALPHA",
  "sku": "DEMO-SKU-A1",
  "product_serial": "DEMO-SN-000001",
  "days_since_purchase": 180,
  "claim_amount_inr": 2600,
  "photo_attached": "Y",
  "partner_inspected": "N",
  "claim_description": "demo unit does not start",
  "inspector_note": "synthetic demonstration record",
  "customer_prior_claims": 1,
  "source": "demo"
}
```

## Test

With the virtual environment active:

```powershell
python tests/smoke_test.py
```

## Repository contents

- `app/`: FastAPI endpoint, local scoring runtime and browser screen.
- `model/demo_model.json`: fictional demonstration coefficients.
- `model/demo_reference.json`: fictional partner and product reference values.
- `tests/`: local smoke test using synthetic inputs.

No external or paid API is required.

