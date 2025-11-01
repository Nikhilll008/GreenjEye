"""
app.py — Minimal, beginner-friendly Flask backend for "image -> disease" flow.

Features:
- POST /detect  : accept image upload (form field 'image'), return disease info JSON.
- GET  /health  : quick health check.
- Built-in mock analyzer (mock_analyze) for local testing without a real API.
- Configurable via .env:
    EXTERNAL_API_URL  - real API URL (optional)
    EXTERNAL_API_KEY  - optional API key
    USE_DB            - 'True' to enable SQLite saving (optional)
    USE_MOCK_API      - 'True' to use local mock (recommended for testing)
- Basic validation, normalization, confidence-threshold decision.
"""

import os
import io
import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import requests

# Optional: enable CORS for local dev (if React runs on a different port)
try:
    from flask_cors import CORS
except Exception:
    CORS = None

# Optional DB (simple SQLite via SQLAlchemy) - keep optional for beginners
try:
    from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
except Exception:
    create_engine = None

load_dotenv()  # loads variables from .env if present

# ---------- Config ----------
EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL", "").strip()  # leave blank if not available
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "").strip()
USE_DB = os.getenv("USE_DB", "False").lower() in ("true", "1", "yes")
USE_MOCK_API = os.getenv("USE_MOCK_API", "True").lower() in ("true", "1", "yes")  # default True for easy testing
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))  # default 0.6

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

app = Flask(__name__)
if CORS:
    CORS(app)

# ---------- Optional DB model ----------
Base = None
SessionLocal = None
if USE_DB and create_engine:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///submissions.db")
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    class Submission(Base):
        __tablename__ = "submissions"
        id = Column(Integer, primary_key=True, index=True)
        filename = Column(String(256))
        plant = Column(String(128))
        disease_name = Column(String(128))
        confidence = Column(Float)
        remedy = Column(Text)
        raw_response = Column(Text)
        created_at = Column(DateTime, default=datetime.datetime.utcnow)

    Base.metadata.create_all(bind=engine)


# ---------- Helpers ----------
def allowed_file(filename: str) -> bool:
    if not filename:
        return False
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT


def validate_image_upload(file_storage):
    """Raise ValueError if invalid."""
    if file_storage is None:
        raise ValueError("No file provided (expected form field 'image').")
    if file_storage.filename == "":
        raise ValueError("Empty filename.")
    if not allowed_file(file_storage.filename):
        raise ValueError(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXT)}")
    # optional size check:
    # Note: file_storage.stream may not report size; we read bytes later and check length

# Mock external analyzer (simulates the API your coordinator mentioned)
def mock_analyze(image_bytes: bytes, filename: str) -> dict:
    """
    Return a simulated API response.
    In real use, the backend would call a real API endpoint instead.
    """
    # For the mock, we return fixed example data. In reality the external API
    # will inspect the image and return dynamic results.
    return {
        "plant": "Tomato",
        "diseases": [
            {
                "name": "Early Blight",
                "confidence": 0.92,
                "description": "Brown circular lesions on lower leaves",
                "remedy": "Remove infected leaves; apply recommended fungicide every 7-10 days"
            },
            {
                "name": "Septoria Leaf Spot",
                "confidence": 0.05,
                "description": "Small grey spots with dark margins",
                "remedy": "Improve air circulation; remove crop debris"
            }
        ],
        "note": "mock response"
    }


def send_to_external_api(image_bytes: bytes, filename: str) -> dict:
    """
    If USE_MOCK_API is True, use the internal mock_analyze.
    Otherwise call EXTERNAL_API_URL with the image file.
    Returns normalized JSON (not raw string).
    Raises RuntimeError on network/API errors.
    """
    if USE_MOCK_API or not EXTERNAL_API_URL:
        # Use mock for testing or if no URL configured
        return mock_analyze(image_bytes, filename)

    files = {"image": (filename, io.BytesIO(image_bytes), "application/octet-stream")}
    headers = {}
    if EXTERNAL_API_KEY:
        headers["Authorization"] = f"Bearer {EXTERNAL_API_KEY}"
    try:
        resp = requests.post(EXTERNAL_API_URL, files=files, headers=headers, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to reach external API: {e}")
    if resp.status_code != 200:
        raise RuntimeError(f"External API returned status {resp.status_code}: {resp.text}")
    try:
        return resp.json()
    except ValueError:
        raise RuntimeError("External API returned non-JSON response")


def normalize_api_response(data: dict) -> dict:
    """
    Convert external API shape into our internal normalized shape:
    { plant: str, diseases: [ {name, confidence, description, remedy}, ... ], raw: {...} }
    """
    plant = data.get("plant") or data.get("crop") or None
    diseases = data.get("diseases") or data.get("predictions") or []
    normalized = {"plant": plant, "diseases": []}
    if not isinstance(diseases, list):
        diseases = []
    for d in diseases:
        name = d.get("name") or d.get("disease") or "Unknown"
        confidence = d.get("confidence") or d.get("score") or None
        try:
            confidence = float(confidence) if confidence is not None else None
        except Exception:
            confidence = None
        normalized["diseases"].append({
            "name": name,
            "confidence": confidence,
            "description": d.get("description") or d.get("symptoms") or "",
            "remedy": d.get("remedy") or d.get("treatment") or ""
        })
    normalized["raw"] = data
    return normalized


def choose_primary_disease(normalized: dict, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """
    Apply the decision rules:
    - If no diseases -> status 'no_prediction'
    - Pick disease with highest confidence.
    - If top confidence >= threshold -> status 'ok'
    - Else -> status 'low_confidence' (still return top but mark it)
    """
    diseases = normalized.get("diseases", [])
    if not diseases:
        return {"status": "no_prediction", "chosen": None}

    # sort by confidence (None -> -1) descending
    diseases_sorted = sorted(diseases, key=lambda x: x.get("confidence") if x.get("confidence") is not None else -1, reverse=True)
    top = diseases_sorted[0]
    conf = top.get("confidence")
    if conf is None:
        return {"status": "unknown_confidence", "chosen": top}
    if conf >= threshold:
        return {"status": "ok", "chosen": top}
    else:
        return {"status": "low_confidence", "chosen": top}


def save_submission_to_db(filename: str, normalized: dict, chosen: dict):
    """Save a record into DB if enabled. Fail silently (log) on error."""
    if not USE_DB or SessionLocal is None:
        return
    try:
        session = SessionLocal()
        submission = Submission(
            filename=filename,
            plant=normalized.get("plant"),
            disease_name=chosen.get("name") if chosen else None,
            confidence=chosen.get("confidence") if chosen else None,
            remedy=chosen.get("remedy") if chosen else None,
            raw_response=str(normalized.get("raw"))
        )
        session.add(submission)
        session.commit()
    except Exception as e:
        app.logger.error("DB save failed: %s", e)
    finally:
        try:
            session.close()
        except Exception:
            pass


# ---------- Routes ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "use_mock_api": USE_MOCK_API,
        "use_db": USE_DB,
    })


@app.route("/detect", methods=["POST"])
def detect():
    """
    Main endpoint. Expects multipart/form-data with key 'image'.
    Returns:
    {
      "plant": "...",
      "decision_status": "ok" | "low_confidence" | "no_prediction" | "unknown_confidence",
      "chosen_disease": {name, confidence, description, remedy} | null,
      "all_diseases": [...],
      "notes": "string"
    }
    """
    try:
        file = request.files.get("image")
        validate_image_upload(file)
        # read bytes to check size and send to API
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "Uploaded file is empty"}), 400
        if len(image_bytes) > MAX_IMAGE_BYTES:
            return jsonify({"error": f"File too large (max {MAX_IMAGE_BYTES} bytes)"}), 400

        # Call external API (or mock)
        api_raw = send_to_external_api(image_bytes, file.filename)
        normalized = normalize_api_response(api_raw)
        decision = choose_primary_disease(normalized)

        # Optionally save
        save_submission_to_db(file.filename, normalized, decision.get("chosen"))

        # Compose response
        notes = ""
        if decision["status"] == "low_confidence":
            notes = f"Top confidence {decision['chosen'].get('confidence')} < threshold {CONFIDENCE_THRESHOLD}. Consider re-uploading clearer images or more angles."
        elif decision["status"] == "no_prediction":
            notes = "Could not identify a disease. Ask farmer for clearer images (close-up, good light)."

        response = {
            "plant": normalized.get("plant"),
            "decision_status": decision["status"],
            "chosen_disease": decision.get("chosen"),
            "all_diseases": normalized.get("diseases", []),
            "notes": notes
        }
        return jsonify(response), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        return jsonify({"error": str(re)}), 502
    except Exception as e:
        app.logger.exception("Unexpected error")
        return jsonify({"error": "Internal server error"}), 500


# ---------- (Optional) small debug endpoint to call mock directly if needed ----------
@app.route("/mock_analyze", methods=["POST"])
def mock_analyze_route():
    # Accepts files same as external API and returns mock JSON
    file = request.files.get("image")
    if file:
        image_bytes = file.read()
    else:
        image_bytes = b""
    return jsonify(mock_analyze(image_bytes, file.filename if file else "no-file")), 200


# ---------- Run ----------
if __name__ == "__main__":
    # Use threaded=True so that if testing by HTTP call to /mock_analyze on the same server it won't block.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True, threaded=True)
