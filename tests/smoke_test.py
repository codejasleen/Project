import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.model_runtime import predict

claim = {
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
    "source": "demo",
}

result = predict(claim)
assert 0 <= result["fraud_score"] <= 1
assert result["review_recommendation"] in {"queue_for_human_review", "do_not_prioritize"}
assert result["model"]["mode"] == "synthetic_public_demo"
assert result["explanation"]["higher_risk_signals"]
assert result["explanation"]["lower_risk_signals"]
assert "not proof of fraud" in result["explanation"]["disclaimer"]

unknown = dict(claim, partner_id="DEMO-PARTNER-UNKNOWN-INPUT", sku="DEMO-SKU-UNKNOWN-INPUT")
unknown_result = predict(unknown)
assert len(unknown_result["warnings"]) == 2

print(json.dumps(result, indent=2))
print("PUBLIC DEMO SMOKE TEST PASSED")

