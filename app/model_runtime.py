import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
MODEL = json.loads((MODEL_DIR / "demo_model.json").read_text(encoding="utf-8"))
REFERENCE = json.loads((MODEL_DIR / "demo_reference.json").read_text(encoding="utf-8"))
PARTNERS = {row["partner_id"]: row for row in REFERENCE["partners"]}
PRODUCTS = {row["sku"]: row for row in REFERENCE["products"]}


def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-max(min(value, 35), -35)))


def _serial_format_issue(value):
    return int(not bool(re.fullmatch(r"DEMO-SN-[0-9]{6}", str(value or "").upper())))


def _statement(feature, value, contribution):
    direction = "higher-risk" if contribution > 0 else "lower-risk"
    if feature == "claim_amount_ratio":
        text = f"The claim amount is {value:.0%} of the fictional product list price"
    elif feature == "purchase_age_years":
        text = f"The synthetic claim was submitted after {value * 365:.0f} days"
    elif feature == "customer_prior_claims":
        text = f"The fictional customer has {int(value)} earlier demo claims"
    elif feature == "partner_recent_claims":
        text = f"Synthetic demo history contains {int(value)} recent claims for this fictional partner"
    elif feature == "missing_photo":
        text = "No photo is attached to the demo claim"
    elif feature == "uninspected":
        text = "The demo claim has no recorded inspection"
    elif feature == "serial_format_issue":
        text = "The fictional serial does not match the documented demo format"
    elif feature == "partner_adjustment":
        text = "The synthetic partner profile affected the demo score"
    elif feature == "sku_adjustment":
        text = "The synthetic product profile affected the demo score"
    else:
        text = None
    return f"{text}; for this prediction, this was a {direction} signal." if text else None


def predict(record):
    warnings = []
    partner = PARTNERS.get(record["partner_id"])
    if partner is None:
        warnings.append("Unknown demo partner_id; neutral synthetic partner values were used.")
        partner = REFERENCE["defaults"]["partner"]

    product = PRODUCTS.get(record["sku"])
    if product is None:
        warnings.append("Unknown demo sku; neutral synthetic product values were used.")
        product = REFERENCE["defaults"]["product"]

    amount = float(record["claim_amount_inr"])
    price = float(product["list_price_inr"])
    features = {
        "claim_amount_ratio": amount / price if price else 0.0,
        "purchase_age_years": float(record["days_since_purchase"]) / 365.0,
        "customer_prior_claims": float(record.get("customer_prior_claims", 0) or 0),
        "partner_recent_claims": float(partner["recent_demo_claims"]),
        "missing_photo": float(str(record.get("photo_attached", "N")).upper() != "Y"),
        "uninspected": float(str(record.get("partner_inspected", "N")).upper() != "Y"),
        "serial_format_issue": float(_serial_format_issue(record.get("product_serial"))),
    }

    contributions = []
    for feature, value in features.items():
        contribution = MODEL["feature_weights"][feature] * value
        if abs(contribution) > 1e-12:
            contributions.append((feature, value, contribution))

    partner_adjustment = MODEL["partner_adjustments"].get(record["partner_id"], 0.0)
    sku_adjustment = MODEL["sku_adjustments"].get(record["sku"], 0.0)
    if partner_adjustment:
        contributions.append(("partner_adjustment", record["partner_id"], partner_adjustment))
    if sku_adjustment:
        contributions.append(("sku_adjustment", record["sku"], sku_adjustment))

    logit = MODEL["intercept"] + sum(item[2] for item in contributions)
    score = _sigmoid(logit)
    higher = sorted((item for item in contributions if item[2] > 0), key=lambda item: item[2], reverse=True)
    lower = sorted((item for item in contributions if item[2] < 0), key=lambda item: item[2])
    higher_statements = [_statement(*item) for item in higher[:2]]
    lower_statements = [_statement(*item) for item in lower[:2]]
    disclaimer = "These synthetic signals explain the demo model's prioritisation; the score is not proof of fraud and must not be used for automatic rejection."
    explanation = {
        "title": "Why this demo claim received this score",
        "higher_risk_signals": [text for text in higher_statements if text],
        "lower_risk_signals": [text for text in lower_statements if text],
        "disclaimer": disclaimer,
    }

    return {
        "fraud_score": round(score, 6),
        "review_recommendation": "queue_for_human_review" if score >= MODEL["review_threshold"] else "do_not_prioritize",
        "reasons": explanation["higher_risk_signals"] + explanation["lower_risk_signals"] + [disclaimer],
        "explanation": explanation,
        "warnings": warnings,
        "model": {
            "name": MODEL["name"],
            "version": MODEL["version"],
            "mode": "synthetic_public_demo",
        },
    }

