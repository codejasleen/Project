from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .model_runtime import predict

app = FastAPI(title="Warranty Review Public Demo", version="1.0.0")
INDEX = Path(__file__).resolve().parent / "static" / "index.html"


class Claim(BaseModel):
    claim_id: str
    submitted_at: str
    partner_id: str
    sku: str
    product_serial: str
    days_since_purchase: int = Field(ge=0)
    claim_amount_inr: float = Field(gt=0)
    photo_attached: str = "N"
    partner_inspected: str = "N"
    claim_description: str = ""
    inspector_note: Optional[str] = ""
    customer_prior_claims: int = Field(default=0, ge=0)
    source: str = "demo"


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(INDEX)


@app.get("/health")
def health():
    return {"status": "ok", "mode": "synthetic_public_demo"}


@app.post("/predict")
def score_claim(claim: Claim):
    return predict(claim.model_dump())

