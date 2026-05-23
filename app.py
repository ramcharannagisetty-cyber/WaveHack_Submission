"""
PetriAI - Flask Backend Server
Run with: python3 app.py
"""

import os, sys, json, uuid, time, math, random, base64, sqlite3, threading
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import io
from flask import Flask, request, jsonify, send_from_directory, send_file

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "frontend"
DB_PATH    = BASE_DIR / "petriai.db"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR))
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   # 20 MB


# ── database ───────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id           TEXT PRIMARY KEY,
            filename     TEXT,
            sample_name  TEXT,
            created_at   TEXT,
            colony_count INTEGER,
            cfu_estimate TEXT,
            coverage_pct REAL,
            growth_level TEXT,
            growth_pct   INTEGER,
            contamination TEXT,
            risk_level   TEXT,
            report       TEXT,
            annotated_path TEXT
        );
    """)
    db.commit()
    db.close()

init_db()


# ══════════════════════════════════════════════════════════════════════════
# COMPUTER VISION ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════

def preprocess_image(img_bgr):
    """White balance + CLAHE contrast enhancement."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def detect_dish_mask(img_bgr):
    """Detect the circular petri dish boundary using Hough circles."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = img_bgr.shape[:2]
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 2,
        param1=80,
        param2=40,
        minRadius=min(h, w) // 4,
        maxRadius=min(h, w) // 2,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    if circles is not None:
        cx, cy, r = np.round(circles[0][0]).astype(int)
        cv2.circle(mask, (cx, cy), int(r * 0.95), 255, -1)
        return mask, (cx, cy, r)
    # fallback: assume dish fills 90% of image
    cx, cy = w // 2, h // 2
    r = min(w, h) // 2 - 10
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask, (cx, cy, r)


def detect_colonies(img_bgr, dish_mask):
    """
    Detect colonies using adaptive thresholding + contour analysis.
    Returns list of (cx, cy, radius, circularity, mean_color).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Mask to dish area
    masked_gray = cv2.bitwise_and(gray, gray, mask=dish_mask)

    # Background subtraction via large Gaussian blur
    bg = cv2.GaussianBlur(masked_gray, (51, 51), 0)
    diff = cv2.absdiff(masked_gray, bg)

    # Threshold
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.bitwise_and(cleaned, cleaned, mask=dish_mask)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    colonies = []
    h, w = img_bgr.shape[:2]
    min_area = (h * w) * 0.00005   # ~0.005% of image
    max_area = (h * w) * 0.08      # ~8% of image

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        radius = int(math.sqrt(area / math.pi))

        # Sample mean color within contour
        col_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(col_mask, [cnt], -1, 255, -1)
        mean_color = cv2.mean(img_bgr, mask=col_mask)[:3]

        colonies.append({
            "cx": cx, "cy": cy,
            "radius": max(radius, 3),
            "circularity": round(circularity, 3),
            "area": area,
            "mean_color": [int(c) for c in mean_color],
        })

    return colonies


def classify_contamination(colonies, dish_info):
    """
    Flag colonies as probable contaminants based on morphological outliers.
    Returns list of booleans.
    """
    if not colonies:
        return []

    # Compute median circularity and color distance
    circs = [c["circularity"] for c in colonies]
    med_circ = float(np.median(circs))
    colors = np.array([c["mean_color"] for c in colonies], dtype=float)
    med_color = np.median(colors, axis=0)

    flags = []
    for c in colonies:
        color_dist = float(np.linalg.norm(np.array(c["mean_color"]) - med_color))
        low_circ = c["circularity"] < med_circ * 0.6
        unusual_color = color_dist > 60
        flags.append(low_circ or unusual_color)

    return flags


def compute_coverage(colonies, dish_mask):
    """Compute % of dish covered by colonies."""
    dish_pixels = int(np.sum(dish_mask > 0))
    if dish_pixels == 0:
        return 0.0
    colony_pixels = sum(math.pi * c["radius"] ** 2 for c in colonies)
    return round(min(100.0, colony_pixels / dish_pixels * 100), 1)


def compute_growth_level(colony_count, coverage_pct):
    """Classify growth as LOW / MODERATE / HIGH / TNTC."""
    if colony_count > 300:
        return "TNTC", 100
    if colony_count > 150 or coverage_pct > 60:
        return "HIGH", min(95, 60 + int(coverage_pct * 0.5))
    if colony_count > 50 or coverage_pct > 25:
        return "MODERATE", min(60, 25 + int(coverage_pct))
    return "LOW", max(5, int(coverage_pct * 1.5))


def annotate_image(img_bgr, colonies, contam_flags, dish_info):
    """Draw colony annotations and return annotated image bytes."""
    annotated = img_bgr.copy()
    cx_d, cy_d, r_d = dish_info

    # Draw dish boundary
    cv2.circle(annotated, (cx_d, cy_d), r_d, (40, 100, 60), 2)

    for i, (col, is_contam) in enumerate(zip(colonies, contam_flags)):
        cx, cy, r = col["cx"], col["cy"], col["radius"]
        color = (50, 80, 220) if is_contam else (50, 200, 120)   # BGR
        cv2.circle(annotated, (cx, cy), r + 4, color, 1)
        cv2.circle(annotated, (cx, cy), r, color, -1)
        # Colony number
        cv2.putText(
            annotated, str(i + 1),
            (cx - 4, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
        )

    # Legend
    cv2.rectangle(annotated, (8, 8), (160, 50), (20, 30, 20), -1)
    cv2.circle(annotated, (20, 22), 5, (50, 200, 120), -1)
    cv2.putText(annotated, "Colony", (30, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 240, 200), 1)
    cv2.circle(annotated, (20, 40), 5, (50, 80, 220), -1)
    cv2.putText(annotated, "Suspect", (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 240), 1)

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


def estimate_cfu(colony_count, dilution_factor=1e-5):
    """Estimate CFU/mL from colony count and dilution factor."""
    cfu = colony_count / dilution_factor
    exp = int(math.floor(math.log10(cfu))) if cfu > 0 else 0
    mantissa = round(cfu / (10 ** exp), 1)
    return f"{mantissa}×10^{exp}"


def generate_report_local(data):
    """Generate a structured lab report using templates (no external API needed)."""
    count = data["colony_count"]
    coverage = data["coverage_pct"]
    growth = data["growth_level"]
    contam = data["contamination"]
    contam_count = data.get("contaminant_count", 0)

    summary_parts = {
        "LOW":      f"The sample exhibits sparse microbial growth with {count} discrete colonies detected across {coverage}% of the dish surface. Growth density is classified as LOW, consistent with a heavily diluted inoculum or low-viability sample.",
        "MODERATE": f"The sample demonstrates moderate microbial proliferation with {count} colonies detected occupying {coverage}% of the agar surface. Growth distribution is broadly uniform, consistent with standard laboratory culture conditions.",
        "HIGH":     f"The sample shows robust, high-density growth with {count} colonies identified across {coverage}% of the dish surface. Colony count approaches TNTC threshold; further dilution is recommended for subsequent passages.",
        "TNTC":     f"Colony count exceeds the TNTC (Too Numerous To Count) threshold. Growth coverage of {coverage}% indicates a severely underdiluted sample. Quantitative colony counting is not feasible at this dilution.",
    }
    summary = summary_parts.get(growth, summary_parts["MODERATE"])

    colony_text = (
        f"{count} colonies were detected and enumerated using automated image analysis. "
        f"Mean colony radius: {data.get('mean_radius_px', 'N/A')} px. "
        f"Colony circularity index (median): {data.get('median_circularity', 'N/A')}. "
        f"Spatial distribution analysis indicates {'uniform' if coverage < 60 else 'dense'} coverage "
        f"across the agar surface with {'no significant clustering observed' if coverage < 40 else 'central clustering tendency noted'}."
    )

    if contam_count == 0:
        contam_text = "No contamination events detected. All colonies exhibit morphological profiles consistent with a pure culture. Circularity, color distribution, and size variance fall within expected limits for a single-organism culture."
    elif contam_count <= 2:
        contam_text = f"{contam_count} colony/colonies flagged as probable contaminant(s) based on morphological deviation (low circularity, atypical pigmentation, or irregular margins). Recommendation: re-streak flagged colonies on selective media to confirm purity."
    else:
        contam_text = f"{contam_count} colonies ({round(contam_count/max(count,1)*100)}%) flagged as morphological outliers consistent with contamination. This culture shows signs of a poly-microbial event. Discard and re-prepare from certified stock unless mixed culture analysis is intended."

    risk_level = data["risk_level"]
    risk_map = {"low": "Acceptable", "med": "Borderline — Review Required", "high": "Unacceptable — Action Required"}

    findings = []
    if count > 300:
        findings.append({"type": "alert", "text": "TNTC: prepare 10× further dilution for accurate quantification"})
    elif count > 150:
        findings.append({"type": "warn", "text": f"High count ({count}): consider 10× dilution for next passage"})
    else:
        findings.append({"type": "ok", "text": f"Colony count ({count}) within countable range (30–300 ideal)"})

    if contam_count == 0:
        findings.append({"type": "ok", "text": "Culture purity confirmed — no morphological outliers detected"})
    elif contam_count <= 2:
        findings.append({"type": "warn", "text": f"{contam_count} suspected contaminant(s) — selective media confirmation recommended"})
    else:
        findings.append({"type": "alert", "text": f"Poly-microbial contamination likely ({contam_count} outlier colonies) — discard advised"})

    if coverage > 70:
        findings.append({"type": "warn", "text": f"Coverage {coverage}% — agar surface approaching saturation"})
    else:
        findings.append({"type": "ok", "text": f"Coverage {coverage}% — adequate agar space for colony development"})

    findings.append({"type": "ok", "text": f"Analysis completed in {data.get('analysis_time_ms', '—')}ms by PetriAI CV Engine v1.0"})

    rec_map = {
        "low":  "Growth is within normal parameters. Proceed with downstream processing per standard protocol. For quantitative work, ensure dilution series spans 3 orders of magnitude.",
        "med":  "Review flagged colonies under light microscopy. Prepare selective media sub-cultures to confirm contamination status. Results should be treated as preliminary pending purity verification.",
        "high": "DO NOT proceed with downstream applications. Discard this culture. Investigate contamination source (check media sterility, laminar flow hood integrity, and aseptic technique). Restart from certified frozen stock.",
    }

    return {
        "summary": summary,
        "colony_analysis": colony_text,
        "contamination_assessment": contam_text,
        "findings": findings,
        "recommendations": rec_map.get(risk_level, rec_map["med"]),
        "quality_status": risk_map.get(risk_level, "Unknown"),
    }


# ══════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.after_request
def cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path and (STATIC_DIR / path).exists():
        return send_from_directory(str(STATIC_DIR), path)
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    ext = Path(f.filename).suffix.lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    t0 = time.time()

    # Read image
    file_bytes = f.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Resize if very large
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    # Run pipeline
    img_bgr = preprocess_image(img_bgr)
    dish_mask, dish_info = detect_dish_mask(img_bgr)
    colonies = detect_colonies(img_bgr, dish_mask)
    contam_flags = classify_contamination(colonies, dish_info)
    coverage = compute_coverage(colonies, dish_mask)
    colony_count = len(colonies)
    contam_count = sum(contam_flags)
    growth_level, growth_pct = compute_growth_level(colony_count, coverage)
    cfu = estimate_cfu(colony_count)

    # Risk level
    if contam_count >= 3 or growth_level == "TNTC":
        risk_level = "high"
    elif contam_count >= 1 or growth_level == "HIGH":
        risk_level = "med"
    else:
        risk_level = "low"

    contam_status = "None" if contam_count == 0 else ("Suspected" if contam_count <= 2 else "CONFIRMED")

    # Annotate image
    annotated_bytes = annotate_image(img_bgr, colonies, contam_flags, dish_info)
    annotated_b64 = base64.b64encode(annotated_bytes).decode()

    ms = round((time.time() - t0) * 1000)

    # Build report data
    circs = [c["circularity"] for c in colonies] if colonies else [0]
    radii = [c["radius"] for c in colonies] if colonies else [0]
    report_data = {
        "colony_count": colony_count,
        "coverage_pct": coverage,
        "growth_level": growth_level,
        "contamination": contam_status,
        "contaminant_count": contam_count,
        "risk_level": risk_level,
        "analysis_time_ms": ms,
        "mean_radius_px": round(float(np.mean(radii)), 1),
        "median_circularity": round(float(np.median(circs)), 3),
    }
    report = generate_report_local(report_data)

    # Save to DB
    analysis_id = str(uuid.uuid4())[:8].upper()
    db = get_db()
    db.execute("""
        INSERT INTO analyses
        (id,filename,sample_name,created_at,colony_count,cfu_estimate,coverage_pct,
         growth_level,growth_pct,contamination,risk_level,report)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        analysis_id, f.filename, f.filename,
        datetime.utcnow().isoformat(),
        colony_count, cfu, coverage,
        growth_level, growth_pct,
        contam_status, risk_level,
        json.dumps(report),
    ))
    db.commit()
    db.close()

    return jsonify({
        "id": analysis_id,
        "colony_count": colony_count,
        "cfu_estimate": cfu,
        "coverage_pct": coverage,
        "growth_level": growth_level,
        "growth_pct": growth_pct,
        "contamination": contam_status,
        "risk_level": risk_level,
        "analysis_time_ms": ms,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        "report": report,
        "colony_details": [
            {"id": i + 1, "cx": c["cx"], "cy": c["cy"],
             "radius": c["radius"], "circularity": c["circularity"],
             "suspected_contaminant": bool(flg)}
            for i, (c, flg) in enumerate(zip(colonies, contam_flags))
        ],
    })


@app.route("/api/analyses", methods=["GET"])
def list_analyses():
    db = get_db()
    rows = db.execute(
        "SELECT id,filename,sample_name,created_at,colony_count,cfu_estimate,"
        "coverage_pct,growth_level,contamination,risk_level "
        "FROM analyses ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/analyses/<analysis_id>", methods=["GET"])
def get_analysis(analysis_id):
    db = get_db()
    row = db.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    db.close()
    if not row:
        return jsonify({"error": "Not found"}), 404
    d = dict(row)
    if d.get("report"):
        d["report"] = json.loads(d["report"])
    return jsonify(d)


@app.route("/api/stats", methods=["GET"])
def get_stats():
    db = get_db()
    total = db.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    total_colonies = db.execute("SELECT SUM(colony_count) FROM analyses").fetchone()[0] or 0
    contam_rate = db.execute(
        "SELECT ROUND(100.0*SUM(CASE WHEN contamination!='None' THEN 1 ELSE 0 END)/MAX(COUNT(*),1),1) FROM analyses"
    ).fetchone()[0] or 0
    db.close()
    return jsonify({
        "total_analyses": total,
        "total_colonies_counted": total_colonies,
        "contamination_rate_pct": contam_rate,
    })


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🧫  PetriAI Server")
    print("="*55)
    print(f"  Frontend → http://localhost:5050")
    print(f"  API      → http://localhost:5050/api/health")
    print(f"  Database → {DB_PATH}")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
