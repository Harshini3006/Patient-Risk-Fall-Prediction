"""
╔══════════════════════════════════════════════════════════════════════╗
║        FallGuard AI – Patient Fall Risk Monitoring System           ║
║                      SINGLE FILE VERSION                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  HOW TO RUN:                                                        ║
║    1. pip install flask scikit-learn pandas numpy                   ║
║    2. Place fall_detection_dataset.csv in the same folder           ║
║    3. python fallguard_app.py                                       ║
║    4. Open http://127.0.0.1:5000                                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 – IMPORTS
# ══════════════════════════════════════════════════════════════════════
import os, sys, sqlite3, pickle, io
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template_string, request,
                   jsonify, redirect, url_for, session)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ══════════════════════════════════════════════════════════════════════
# SECTION 2 – APP CONFIG & CONSTANTS
# ══════════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = "fallguard_secret_key_2024"

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "fallguard.db")
MODEL_PATH = os.path.join(BASE_DIR, "fallguard_model.pkl")
CSV_PATH   = os.path.join(BASE_DIR, "fall_detection_dataset.csv")

# Hardcoded users – extend as needed
USERS = {
    "dr.smith":   {"password": "doctor123", "role": "Doctor", "name": "Dr. Smith"},
    "dr.jones":   {"password": "doctor456", "role": "Doctor", "name": "Dr. Jones"},
    "nurse.anna": {"password": "nurse123",  "role": "Nurse",  "name": "Anna Wilson"},
    "nurse.bob":  {"password": "nurse456",  "role": "Nurse",  "name": "Bob Carter"},
}

# ══════════════════════════════════════════════════════════════════════
# SECTION 3 – DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════
def init_db():
    """Create tables and seed demo patients if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        username  TEXT UNIQUE NOT NULL,
        password  TEXT NOT NULL,
        name      TEXT NOT NULL,
        role      TEXT NOT NULL,
        status    TEXT DEFAULT 'Active',
        created_at TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS patients (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id  TEXT UNIQUE NOT NULL,
        name        TEXT NOT NULL,
        age         INTEGER,
        room_type   TEXT,
        assigned_to TEXT,
        created_at  TEXT
    )""")

    # Add assigned_by column if it doesn't exist
    try:
        c.execute("ALTER TABLE patients ADD COLUMN assigned_by TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id    TEXT,
        acc_x REAL, acc_y REAL, acc_z REAL,
        gyro_x REAL, gyro_y REAL, gyro_z REAL,
        heart_rate    REAL,
        room_temp     REAL,
        room_light    REAL,
        room_type     TEXT,
        posture       TEXT,
        time_of_day   TEXT,
        risk_level    TEXT,
        fall_event    INTEGER,
        fall_severity TEXT,
        risk_prob     REAL,
        checked       INTEGER DEFAULT 0,
        timestamp     TEXT
    )""")

    # Seed users (admin + team)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    demo_users = [
        ("admin", "admin123", "Administrator", "Admin", "Active"),
        ("dr.smith", "doctor123", "Dr. Smith", "Doctor", "Active"),
        ("dr.jones", "doctor456", "Dr. Jones", "Doctor", "Active"),
        ("nurse.anna", "nurse123", "Anna Wilson", "Nurse", "Active"),
        ("nurse.bob", "nurse456", "Bob Carter", "Nurse", "Active"),
    ]
    for user in demo_users:
        c.execute("""INSERT OR IGNORE INTO users
            (username,password,name,role,status,created_at)
            VALUES (?,?,?,?,?,?)""", (*user, ts))

    # Seed 4 demo patients
    demo = [
        ("P001","Alice Johnson", 72,"Bedroom",   "nurse.anna", "dr.smith"),
        ("P002","Robert Brown",  81,"Bathroom",  "nurse.anna", "dr.smith"),
        ("P003","Mary Davis",    68,"LivingRoom","nurse.bob",  "dr.jones"),
        ("P004","James Wilson",  90,"Hallway",   "nurse.bob",  "dr.jones"),
    ]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for d in demo:
        c.execute("""INSERT OR IGNORE INTO patients
            (patient_id,name,age,room_type,assigned_to,assigned_by,created_at)
            VALUES (?,?,?,?,?,?,?)""", (*d, ts))

    # Seed demo predictions to populate the dashboard (only if table is empty)
    c.execute("SELECT COUNT(*) FROM predictions")
    if c.fetchone()[0] == 0:
        demo_preds = [
            ("P001", 0.5, 0.3, 0.2, 0.1, 0.05, 0.08, 78, 22.5, 350, "Bedroom", "Standing", "Morning", "Low", 0, None, 0.15),
            ("P002", 0.8, 0.9, 0.7, 0.3, 0.2, 0.1, 82, 23.0, 200, "Bathroom", "Lying", "Evening", "High", 1, "Moderate", 0.85),
            ("P003", 0.2, 0.1, 0.3, 0.05, 0.08, 0.02, 75, 21.8, 400, "LivingRoom", "Walking", "Afternoon", "Low", 0, None, 0.12),
            ("P004", 0.6, 0.5, 0.4, 0.2, 0.15, 0.1, 88, 22.2, 300, "Hallway", "Transition", "Night", "Medium", 0, None, 0.55),
        ]
        for pred in demo_preds:
            c.execute("""INSERT INTO predictions
                (patient_id,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,
                 heart_rate,room_temp,room_light,room_type,posture,time_of_day,
                 risk_level,fall_event,fall_severity,risk_prob,timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (*pred, ts))

    conn.commit(); conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ══════════════════════════════════════════════════════════════════════
# SECTION 4 – MACHINE LEARNING: TRAIN & SAVE MODEL
# ══════════════════════════════════════════════════════════════════════
def train_and_save_model():
    """Train Random Forest on CSV, save to MODEL_PATH."""
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Dataset not found: {CSV_PATH}")
        sys.exit(1)

    print("[ML] Training model on fall_detection_dataset.csv ...")
    df = pd.read_csv(CSV_PATH)
    df['fall_severity'] = df['fall_severity'].fillna('None')

    le_room    = LabelEncoder()
    le_posture = LabelEncoder()
    le_tod     = LabelEncoder()
    le_risk    = LabelEncoder()
    le_sev     = LabelEncoder()

    df['room_type_enc']   = le_room.fit_transform(df['room_type'])
    df['posture_enc']     = le_posture.fit_transform(df['posture'])
    df['time_of_day_enc'] = le_tod.fit_transform(df['time_of_day'])

    features = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z',
                'heart_rate','room_temp','room_light',
                'room_type_enc','posture_enc','time_of_day_enc']

    X      = df[features]
    y_risk = le_risk.fit_transform(df['risk_level'])
    y_fall = df['fall_event'].values
    y_sev  = le_sev.fit_transform(df['fall_severity'])

    X_tr, X_te, yr_tr, yr_te = train_test_split(X, y_risk, test_size=0.2, random_state=42)

    clf_risk = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_risk.fit(X_tr, yr_tr)
    print(f"[ML] Risk accuracy : {accuracy_score(yr_te, clf_risk.predict(X_te)):.2%}")

    clf_fall = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_fall.fit(X_tr, y_fall[X_tr.index])
    print(f"[ML] Fall accuracy : {accuracy_score(y_fall[X_te.index], clf_fall.predict(X_te)):.2%}")

    clf_sev = DecisionTreeClassifier(random_state=42)
    clf_sev.fit(X_tr, y_sev[X_tr.index])

    bundle = dict(clf_risk=clf_risk, clf_fall=clf_fall, clf_sev=clf_sev,
                  le_room=le_room, le_posture=le_posture, le_tod=le_tod,
                  le_risk=le_risk, le_sev=le_sev, features=features)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[ML] Model saved → {MODEL_PATH}")
    return bundle

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH,'rb') as f:
            return pickle.load(f)
    return train_and_save_model()

MODEL = load_model()

# ══════════════════════════════════════════════════════════════════════
# SECTION 5 – PREDICTION HELPER
# ══════════════════════════════════════════════════════════════════════
def run_prediction(d: dict) -> dict:
    """Run all three classifiers and return a results dict."""
    m = MODEL
    room_enc    = m['le_room'].transform([d['room_type']])[0]
    posture_enc = m['le_posture'].transform([d['posture']])[0]
    tod_enc     = m['le_tod'].transform([d['time_of_day']])[0]

    X = np.array([[d['acc_x'], d['acc_y'], d['acc_z'],
                   d['gyro_x'], d['gyro_y'], d['gyro_z'],
                   d['heart_rate'], d['room_temp'], d['room_light'],
                   room_enc, posture_enc, tod_enc]])

    risk_idx  = m['clf_risk'].predict(X)[0]
    risk_prob = float(max(m['clf_risk'].predict_proba(X)[0]))
    risk_lbl  = m['le_risk'].inverse_transform([risk_idx])[0]

    fall_ev   = int(m['clf_fall'].predict(X)[0])

    sev_idx   = m['clf_sev'].predict(X)[0]
    sev_lbl   = m['le_sev'].inverse_transform([sev_idx])[0]

    # Plain-English explanation
    reasons = []
    if abs(d['acc_x']) > 1.5 or abs(d['acc_y']) > 1.5:
        reasons.append("abnormal lateral acceleration")
    if d['acc_z'] < 8:
        reasons.append("low vertical acceleration (possible fall position)")
    if any(abs(d[k]) > 1.5 for k in ('gyro_x','gyro_y','gyro_z')):
        reasons.append("high gyroscope rotation values")
    if d['heart_rate'] > 100:
        reasons.append("elevated heart rate")
    if d['heart_rate'] < 50:
        reasons.append("very low heart rate")
    if d['posture'] in ('Lying','Transition'):
        reasons.append(f"risky posture ({d['posture']})")
    if d['room_type'] == 'Bathroom':
        reasons.append("high-risk room (Bathroom)")
    if d['time_of_day'] == 'Night':
        reasons.append("night time (reduced visibility)")

    explanation = (f"Risk flagged due to: {', '.join(reasons)}."
                   if reasons else "Sensor readings are within normal range.")

    return dict(risk_level=risk_lbl,
                risk_prob=round(risk_prob*100, 1),
                fall_event=fall_ev,
                fall_severity=sev_lbl if fall_ev else "None",
                explanation=explanation)

# ══════════════════════════════════════════════════════════════════════
# SECTION 6 – AUTH DECORATORS
# ══════════════════════════════════════════════════════════════════════
def login_required(f):
    @wraps(f)
    def decorated(*a, **kw):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*a, **kw)
    return decorated

def doctor_required(f):
    @wraps(f)
    def decorated(*a, **kw):
        if session.get('role') != 'Doctor':
            return redirect(url_for('dashboard'))
        return f(*a, **kw)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*a, **kw):
        if session.get('role') != 'Admin':
            return redirect(url_for('dashboard'))
        return f(*a, **kw)
    return decorated

# ══════════════════════════════════════════════════════════════════════
# SECTION 7 – HTML / CSS / JS  (all inlined as Python strings)
# ══════════════════════════════════════════════════════════════════════

# ── 7a  Global CSS ────────────────────────────────────────────────────
CSS = """
<style>
:root{
  --b7:#1d4ed8;--b6:#2563eb;--b5:#3b82f6;--b1:#dbeafe;--b0:#eff6ff;
  --r6:#dc2626;--r5:#ef4444;--r1:#fee2e2;
  --o5:#f97316;--o1:#ffedd5;
  --g6:#16a34a;--g5:#22c55e;--g1:#dcfce7;
  --gr7:#374151;--gr5:#6b7280;--gr3:#d1d5db;--gr1:#f3f4f6;
  --wh:#fff;--sh:0 1px 4px rgba(0,0,0,.10);--shm:0 4px 16px rgba(0,0,0,.12);
  --rad:10px;--navh:60px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:var(--b0);color:var(--gr7);font-size:14px;line-height:1.5}

/* ── Login ── */
.login-body{display:flex;align-items:center;justify-content:center;min-height:100vh;
  background:linear-gradient(135deg,#1e3a8a 0%,#2563eb 60%,#38bdf8 100%)}
.login-box{background:var(--wh);border-radius:16px;padding:40px 36px;
  width:420px;max-width:96vw;box-shadow:var(--shm)}
.brand-wrap{text-align:center;margin-bottom:28px}
.brand-icon{width:64px;height:64px;background:var(--b6);border-radius:50%;
  display:inline-flex;align-items:center;justify-content:center;margin-bottom:12px}
.brand-icon i{color:#fff;font-size:28px}
.brand-wrap h1{font-size:24px;font-weight:700;color:var(--b7)}
.brand-wrap p{font-size:13px;color:var(--gr5);margin-top:4px}
.demo-hint{background:var(--b0);border:1px solid var(--b1);border-radius:8px;
  padding:12px 14px;font-size:12px;margin-top:16px}
.demo-hint code{background:#e0e7ff;padding:2px 6px;border-radius:4px;font-size:11px}
.forgot-link{text-align:center;margin-top:12px}
.forgot-link a{color:var(--b6);font-size:13px;text-decoration:none}

/* ── Nav ── */
.topnav{position:fixed;top:0;left:0;right:0;height:var(--navh);
  background:var(--b7);color:#fff;display:flex;align-items:center;
  padding:0 20px;gap:16px;z-index:100;box-shadow:0 2px 8px rgba(0,0,0,.2)}
.nav-brand{font-size:18px;font-weight:700;display:flex;align-items:center;gap:8px;white-space:nowrap}
.nav-brand i{color:#93c5fd}
.nav-links{display:flex;gap:4px;flex:1}
.nav-link{color:rgba(255,255,255,.75);text-decoration:none;padding:6px 12px;
  border-radius:6px;font-size:13px;display:flex;align-items:center;gap:6px;transition:.2s}
.nav-link:hover,.nav-link.active{background:rgba(255,255,255,.15);color:#fff}
.nav-user{display:flex;align-items:center;gap:10px;margin-left:auto;white-space:nowrap}
.role-badge{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600}
.role-doctor{background:#7c3aed;color:#fff}.role-nurse{background:#059669;color:#fff}
.user-name{font-size:13px;color:rgba(255,255,255,.85)}
.btn-logout{background:rgba(255,255,255,.12);color:#fff;padding:5px 12px;
  border-radius:6px;text-decoration:none;font-size:12px;transition:.2s}
.btn-logout:hover{background:rgba(255,255,255,.22)}

/* ── Layout ── */
.page-wrap{margin-top:var(--navh);padding:24px;max-width:1400px;margin-left:auto;margin-right:auto}
.page-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
.page-hdr h2{font-size:20px;font-weight:700;color:var(--b7);display:flex;align-items:center;gap:8px}
.subtitle{color:var(--gr5);font-size:13px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}
.two-col-sm{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.three-col{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
.four-col{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:20px}
@media(max-width:900px){.two-col,.four-col{grid-template-columns:1fr 1fr}}
@media(max-width:600px){.two-col,.four-col,.three-col{grid-template-columns:1fr}}

/* ── Cards ── */
.card{background:var(--wh);border-radius:var(--rad);padding:20px;
  display:flex;align-items:center;gap:16px;box-shadow:var(--sh);border-left:4px solid transparent}
.card-blue{border-color:var(--b6)}.card-red{border-color:var(--r5)}
.card-orange{border-color:var(--o5)}.card-green{border-color:var(--g5)}
.card-icon{width:48px;height:48px;border-radius:10px;display:flex;
  align-items:center;justify-content:center;font-size:20px;flex-shrink:0}
.card-blue .card-icon{background:var(--b1);color:var(--b6)}
.card-red .card-icon{background:var(--r1);color:var(--r6)}
.card-orange .card-icon{background:var(--o1);color:var(--o5)}
.card-green .card-icon{background:var(--g1);color:var(--g6)}
.card-num{display:block;font-size:28px;font-weight:700;line-height:1}
.card-lbl{display:block;font-size:12px;color:var(--gr5);margin-top:4px}

/* ── Panel ── */
.panel{background:var(--wh);border-radius:var(--rad);box-shadow:var(--sh);margin-bottom:20px;overflow:hidden}
.panel-hdr{padding:14px 20px;border-bottom:1px solid var(--gr1);font-weight:600;
  font-size:14px;color:var(--b7);display:flex;align-items:center;gap:8px;justify-content:space-between}
.panel-hdr-alert{background:#fef2f2;border-color:var(--r1);color:var(--r6)}
.panel-body{padding:20px}

/* ── Forms ── */
.form-group{margin-bottom:14px}
.form-group label{display:block;font-size:12px;font-weight:600;color:var(--gr7);margin-bottom:5px}
.fc{width:100%;padding:9px 12px;border:1px solid var(--gr3);border-radius:7px;
  font-size:14px;color:var(--gr7);background:var(--wh);transition:.2s;outline:none}
.fc:focus{border-color:var(--b5);box-shadow:0 0 0 3px rgba(37,99,235,.1)}
.pwd-wrap{position:relative}
.pwd-wrap .fc{padding-right:42px}
.toggle-pwd{position:absolute;right:10px;top:50%;transform:translateY(-50%);
  background:none;border:none;cursor:pointer;color:var(--gr5);padding:4px}
.req{color:var(--r5)}
.align-end{display:flex;align-items:flex-end;gap:8px}
.sensor-lbl{font-size:11px;font-weight:700;color:var(--b6);
  text-transform:uppercase;letter-spacing:.5px;margin:16px 0 8px}

/* ── Buttons ── */
.btn{display:inline-flex;align-items:center;gap:6px;padding:9px 18px;border-radius:7px;
  font-size:13px;font-weight:600;cursor:pointer;border:none;text-decoration:none;transition:.2s}
.btn-p{background:var(--b6);color:#fff}.btn-p:hover{background:var(--b7)}
.btn-s{background:var(--gr1);color:var(--gr7);border:1px solid var(--gr3)}.btn-s:hover{background:var(--gr3)}
.btn-d{background:var(--r6);color:#fff}.btn-d:hover{background:#b91c1c}
.btn-full{width:100%;justify-content:center;margin-top:8px}
.btn-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
.sm-btn{padding:5px 10px;font-size:12px;border-radius:5px;display:inline-flex;
  align-items:center;gap:4px;text-decoration:none;cursor:pointer;font-weight:600;border:none}
.sm-blue{background:var(--b1);color:var(--b7)}
.sm-green{background:var(--g1);color:var(--g6)}
.sm-grey{background:var(--gr1);color:var(--gr7)}

/* ── Badges / Alerts ── */
.alert{padding:10px 14px;border-radius:7px;margin-bottom:12px;
  display:flex;align-items:center;gap:8px;font-size:13px}
.alert-err{background:var(--r1);color:var(--r6);border:1px solid #fca5a5}
.alert-ok{background:var(--g1);color:var(--g6);border:1px solid #86efac}
.badge{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;
  border-radius:20px;font-size:11px;font-weight:600}
.bh{background:var(--r1);color:var(--r6)}.bm{background:var(--o1);color:var(--o5)}
.bl{background:var(--g1);color:var(--g6)}.br{background:var(--r1);color:var(--r6)}
.bg{background:var(--g1);color:var(--g6)}.bb{background:var(--b1);color:var(--b6)}
.bgr{background:var(--gr1);color:var(--gr5)}

/* ── Table ── */
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl th{background:var(--gr1);padding:10px 12px;text-align:left;font-weight:600;
  font-size:12px;color:var(--gr5);border-bottom:1px solid var(--gr3)}
.tbl td{padding:10px 12px;border-bottom:1px solid var(--gr1);vertical-align:middle}
.tbl tr:last-child td{border-bottom:none}
.tbl tr:hover td{background:var(--b0)}
.row-alert td{background:#fff5f5!important}
.tbl-wrap{overflow-x:auto}

/* ── Patient grid (Nurse) ── */
.pt-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
.pt-card{border:1px solid var(--gr3);border-radius:var(--rad);padding:16px;
  display:flex;flex-direction:column;gap:10px;background:var(--b0)}
.pt-avatar{width:40px;height:40px;background:var(--b1);border-radius:50%;
  display:flex;align-items:center;justify-content:center;color:var(--b6);font-size:18px}
.pt-info h4{font-size:15px;font-weight:600}
.pt-info p{font-size:12px;color:var(--gr5);margin-top:2px}
.pt-actions{display:flex;gap:8px}

/* ── Alert card (Nurse) ── */
.al-card{background:#fff5f5;border:1px solid #fca5a5;border-radius:var(--rad);
  padding:14px 16px;margin-bottom:10px;display:flex;
  align-items:center;justify-content:space-between;gap:12px}
.al-left{display:flex;align-items:center;gap:12px}
.al-icon{width:40px;height:40px;background:var(--r1);border-radius:50%;
  display:flex;align-items:center;justify-content:center;color:var(--r6);font-size:18px;flex-shrink:0}
.al-right{text-align:right;display:flex;flex-direction:column;gap:4px;align-items:flex-end}

/* ── Recs ── */
.recs{display:flex;flex-direction:column;gap:12px}
.rec{display:flex;align-items:flex-start;gap:12px;padding:14px 16px;border-radius:var(--rad);font-size:13px}
.rec i{font-size:18px;margin-top:2px;flex-shrink:0}
.rec-r{background:var(--r1);color:var(--r6)}
.rec-o{background:var(--o1);color:#c2410c}
.rec-g{background:var(--g1);color:var(--g6)}

/* ── Sensor layout ── */
.sensor-wrap{display:grid;grid-template-columns:1fr 420px;gap:20px;align-items:start}
@media(max-width:900px){.sensor-wrap{grid-template-columns:1fr}}

/* ── Prediction result ── */
.res-patient{display:flex;align-items:center;gap:12px;margin-bottom:16px}
.res-patient i{font-size:28px;color:var(--b5)}
.risk-card{border-radius:var(--rad);padding:20px;text-align:center;margin-bottom:16px}
.risk-r{background:var(--r1);border:2px solid var(--r5)}
.risk-o{background:var(--o1);border:2px solid var(--o5)}
.risk-g{background:var(--g1);border:2px solid var(--g5)}
.risk-lbl{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;opacity:.7}
.risk-val{font-size:32px;font-weight:800;margin:4px 0}
.risk-prob{font-size:12px;opacity:.7;margin-bottom:8px}
.prob-bar{height:8px;background:rgba(0,0,0,.1);border-radius:4px;overflow:hidden}
.prob-fill{height:100%;background:currentColor;border-radius:4px;transition:width .5s}
.res-badges{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.res-badge{padding:6px 12px;border-radius:7px;font-size:12px;font-weight:600;
  display:inline-flex;align-items:center;gap:6px}
.explain{background:var(--b0);border-radius:var(--rad);padding:12px 14px;
  margin-bottom:12px;font-size:13px;border-left:3px solid var(--b5)}
.explain p{margin-top:6px;color:var(--gr7)}
.recomm{border-radius:var(--rad);padding:12px 14px;font-size:13px}
.recomm p{margin-top:6px}

/* ── Charts ── */
.chart-wrap{position:relative;height:260px}
.chart-wrap canvas{max-height:100%}

/* ── Filter row ── */
.filter-row{display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:12px;align-items:end}
@media(max-width:700px){.filter-row{grid-template-columns:1fr 1fr}}

/* ── Modal ── */
.modal-ov{position:fixed;inset:0;background:rgba(0,0,0,.5);
  display:flex;align-items:center;justify-content:center;z-index:999}
.modal-card{background:var(--wh);border-radius:14px;width:460px;max-width:96vw;
  box-shadow:var(--shm);overflow:hidden}
.modal-hdr{padding:16px 20px;background:var(--b6);color:#fff;font-weight:700;
  font-size:15px;display:flex;align-items:center;justify-content:space-between}
.modal-hdr-alert{background:var(--r6)}
.modal-cls{background:none;border:none;color:rgba(255,255,255,.8);font-size:18px;cursor:pointer;line-height:1}
.modal-body{padding:20px}
.modal-ft{padding:14px 20px;border-top:1px solid var(--gr1);display:flex;gap:10px;justify-content:flex-end}
.alert-pulse{font-size:56px;margin-bottom:12px;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.1)}}
.text-center{text-align:center}

/* ── Toast ── */
#toast-wrap{position:fixed;bottom:20px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:8px}
.toast{padding:12px 18px;border-radius:8px;font-size:13px;font-weight:500;
  box-shadow:var(--shm);animation:slideIn .3s ease;max-width:320px}
.toast-ok{background:var(--g6);color:#fff}
.toast-err{background:var(--r6);color:#fff}
.toast-info{background:var(--b6);color:#fff}
@keyframes slideIn{from{transform:translateX(120%);opacity:0}to{transform:translateX(0);opacity:1}}

/* ── Admin Dashboard ── */
.metric-card{background:var(--wh);border-radius:12px;padding:20px;
  display:flex;gap:16px;align-items:center;box-shadow:var(--sh)}
.metric-icon{width:60px;height:60px;border-radius:10px;
  display:flex;align-items:center;justify-content:center;font-size:24px;color:#fff}
.bg-blue{background:var(--b6)}
.bg-green{background:var(--g6)}
.bg-purple{background:#a855f7}
.bg-orange{background:var(--o6)}
.metric-info{flex:1}
.metric-label{font-size:12px;color:var(--gr5);margin-bottom:4px}
.metric-value{font-size:28px;font-weight:700;color:var(--b7)}
.action-cell{display:flex;gap:8px}
.badge{display:inline-block;padding:4px 8px;border-radius:4px;font-size:12px;font-weight:600}
.bd{background:#dbeafe;color:#1e40af}
.bn{background:#e9d5ff;color:#6b21a8}
.br{background:#fee2e2;color:#991b1b}
.by{background:#fef3c7;color:#92400e}
.bg{background:#dcfce7;color:#166534}
.bs-ok{background:#dcfce7;color:#166534}
.bs-warn{background:#fef3c7;color:#92400e}
.row-alert{background:#fef2f2;border-left:3px solid #ef4444}
.sm-btn{padding:6px 12px;font-size:12px;border:none;border-radius:6px;
  cursor:pointer;display:inline-flex;gap:6px;align-items:center;font-weight:600;transition:all .2s}
.sm-blue{background:#dbeafe;color:#1e40af}
.sm-blue:hover{background:#bfdbfe;transform:translateY(-1px)}
.sm-red{background:#fee2e2;color:#991b1b}
.sm-red:hover{background:#fecaca;transform:translateY(-1px)}
.three-col{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px}

/* ── Misc ── */
.muted{color:var(--gr5);font-size:12px}
.empty{text-align:center;padding:32px;color:var(--gr5);font-size:14px}
.loading-row{text-align:center;padding:24px;color:var(--gr5)}
</style>
"""

# ── 7b  Nav snippet ────────────────────────────────────────────────────
def nav(active=""):
    role = session.get("role","")
    name = session.get("name","")
    links = []
    if role == "Admin":
        links = [("/admin","fa-crown","Dashboard")]
    elif role == "Doctor":
        links = [("/doctor","fa-th-large","Dashboard"),
                 ("/sensor","fa-microchip","Sensor Input"),
                 ("/history","fa-history","History"),
                 ("/analytics","fa-chart-bar","Analytics")]
    else:
        links = [("/nurse","fa-th-large","Dashboard"),
                 ("/sensor","fa-microchip","Sensor Input"),
                 ("/history","fa-history","History")]

    li = "".join(
        f'<a href="{href}" class="nav-link {"active" if href==active else ""}">'
        f'<i class="fas {ico}"></i>{lbl}</a>'
        for href,ico,lbl in links
    )
    rb = f'<span class="role-badge role-{role.lower()}">{role}</span>'
    return f"""
    <nav class="topnav">
      <div class="nav-brand"><i class="fas fa-heartbeat"></i> FallGuard AI</div>
      <div class="nav-links">{li}</div>
      <div class="nav-user">
        {rb}
        <span class="user-name"><i class="fas fa-user-circle"></i> {name}</span>
        <a href="/logout" class="btn-logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
      </div>
    </nav>
    <div id="toast-wrap"></div>
    """

# ── 7c  Page shell ─────────────────────────────────────────────────────
def page(title, body, active="", extra_head=""):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} – FallGuard AI</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
{CSS}
{extra_head}
</head>
<body>
{nav(active)}
<div class="page-wrap">{body}</div>
<script>
function showToast(msg,type='info',dur=3500){{
  const w=document.getElementById('toast-wrap');
  if(!w)return;
  const t=document.createElement('div');
  const ico=type==='ok'?'check-circle':type==='err'?'exclamation-circle':'info-circle';
  t.className='toast toast-'+type;
  t.innerHTML='<i class="fas fa-'+ico+'"></i> '+msg;
  w.appendChild(t);
  setTimeout(()=>{{t.style.opacity='0';t.style.transition='opacity .3s';
    setTimeout(()=>t.remove(),350);}},dur);
}}
</script>
</body></html>"""

# ══════════════════════════════════════════════════════════════════════
# SECTION 8 – ROUTES: AUTH
# ══════════════════════════════════════════════════════════════════════
@app.route("/", methods=["GET","POST"])
def login():
    error = ""
    if request.method == "POST":
        u = request.form.get("username","").strip()
        p = request.form.get("password","").strip()
        
        # Check database users first
        db = get_db()
        db_user = db.execute("SELECT id,username,password,name,role FROM users WHERE username=?", (u,)).fetchone()
        db.close()
        
        if db_user and db_user["password"] == p:
            session.update(username=u, role=db_user["role"], name=db_user["name"])
            return redirect(url_for("dashboard"))
        
        # Fall back to hardcoded users
        usr = USERS.get(u)
        if usr and usr["password"] == p:
            session.update(username=u, role=usr["role"], name=usr["name"])
            return redirect(url_for("dashboard"))
        
        error = "Invalid username or password."

    err_html = (f'<div class="alert alert-err"><i class="fas fa-exclamation-circle"></i> {error}</div>'
                if error else "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Login – FallGuard AI</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
{CSS}
</head>
<body class="login-body">
<div class="login-box">
  <div class="brand-wrap">
    <div class="brand-icon"><i class="fas fa-heartbeat"></i></div>
    <h1>FallGuard AI</h1>
    <p>Patient Fall Risk Monitoring System</p>
  </div>
  {err_html}
  <form method="POST" action="/">
    <div class="form-group">
      <label><i class="fas fa-user-tag"></i> Select Role</label>
      <select id="roleHint" class="fc" onchange="hintRole()">
        <option value="">-- Choose your role --</option>
        <option value="Admin">Administrator</option>
        <option value="Doctor">Doctor</option>
        <option value="Nurse">Nurse</option>
      </select>
    </div>
    <div class="form-group">
      <label><i class="fas fa-user"></i> Username</label>
      <input type="text" name="username" id="uname" class="fc" placeholder="Enter username" required>
    </div>
    <div class="form-group">
      <label><i class="fas fa-lock"></i> Password</label>
      <div class="pwd-wrap">
        <input type="password" name="password" id="pwd" class="fc" placeholder="Enter password" required>
        <button type="button" class="toggle-pwd" onclick="togglePwd('pwd','eye1')">
          <i id="eye1" class="fas fa-eye"></i>
        </button>
      </div>
    </div>
    <button type="submit" class="btn btn-p btn-full">
      <i class="fas fa-sign-in-alt"></i> Login
    </button>
    <div class="forgot-link">
      <a href="#" onclick="document.getElementById('fModal').style.display='flex'">
        <i class="fas fa-key"></i> Forgot Password?
      </a>
    </div>
  </form>
</div>

<!-- Forgot Password Modal -->
<div id="fModal" class="modal-ov" style="display:none">
  <div class="modal-card">
    <div class="modal-hdr">
      <span><i class="fas fa-key"></i> Reset Password</span>
      <button class="modal-cls" onclick="closeForgot()">✕</button>
    </div>
    <div class="modal-body">
      <div id="fMsg" class="alert" style="display:none"></div>
      <div class="form-group"><label>Username</label>
        <input type="text" id="fUser" class="fc" placeholder="Your username"></div>
      <div class="form-group"><label>New Password</label>
        <div class="pwd-wrap">
          <input type="password" id="fPwd" class="fc" placeholder="New password">
          <button type="button" class="toggle-pwd" onclick="togglePwd('fPwd','eye2')">
            <i id="eye2" class="fas fa-eye"></i>
          </button>
        </div>
      </div>
      <div class="form-group"><label>Confirm Password</label>
        <input type="password" id="fConf" class="fc" placeholder="Confirm password"></div>
    </div>
    <div class="modal-ft">
      <button class="btn btn-s" onclick="closeForgot()">Cancel</button>
      <button class="btn btn-p" onclick="submitForgot()">
        <i class="fas fa-save"></i> Reset Password
      </button>
    </div>
  </div>
</div>

<script>
function hintRole(){{
  const r=document.getElementById('roleHint').value;
  document.getElementById('uname').placeholder=
    r==='Admin'?'e.g. admin':r==='Doctor'?'e.g. dr.smith':r==='Nurse'?'e.g. nurse.anna':'Enter username';
}}
function togglePwd(id,ico){{
  const i=document.getElementById(id),e=document.getElementById(ico);
  if(i.type==='password'){{i.type='text';e.className='fas fa-eye-slash'}}
  else{{i.type='password';e.className='fas fa-eye'}}
}}
function closeForgot(){{
  document.getElementById('fModal').style.display='none';
  ['fUser','fPwd','fConf'].forEach(x=>document.getElementById(x).value='');
  const m=document.getElementById('fMsg');m.style.display='none';
}}
async function submitForgot(){{
  const u=document.getElementById('fUser').value.trim();
  const p=document.getElementById('fPwd').value;
  const c=document.getElementById('fConf').value;
  const m=document.getElementById('fMsg');
  if(!u||!p){{showFMsg('err','Please fill in all fields.');return;}}
  if(p!==c){{showFMsg('err','Passwords do not match.');return;}}
  if(p.length<6){{showFMsg('err','Password must be at least 6 characters.');return;}}
  const res=await fetch('/forgot-password',{{method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{username:u,new_password:p}})}});
  const data=await res.json();
  if(data.success){{showFMsg('ok',data.message);setTimeout(closeForgot,2000)}}
  else showFMsg('err',data.message);
}}
function showFMsg(type,text){{
  const m=document.getElementById('fMsg');
  m.className=type==='ok'?'alert alert-ok':'alert alert-err';
  m.innerHTML='<i class="fas fa-'+(type==='ok'?'check-circle':'exclamation-circle')+'"></i> '+text;
  m.style.display='flex';
}}
</script>
</body></html>"""
    return html

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    d = request.json
    u = d.get("username","").strip()
    p = d.get("new_password","").strip()
    if u in USERS:
        USERS[u]["password"] = p
        return jsonify({"success": True,  "message": "Password updated successfully!"})
    return jsonify({"success": False, "message": "Username not found."})

@app.route("/dashboard")
@login_required
def dashboard():
    role = session["role"]
    if role == "Admin":
        return redirect(url_for("admin_dashboard"))
    elif role == "Doctor":
        return redirect(url_for("doctor"))
    else:
        return redirect(url_for("nurse"))

# ══════════════════════════════════════════════════════════════════════
# SECTION 9 – ROUTE: DOCTOR DASHBOARD
# ══════════════════════════════════════════════════════════════════════
@app.route("/doctor")
@login_required
@doctor_required
def doctor():
    db = get_db()
    patients = db.execute("SELECT * FROM patients").fetchall()
    preds    = db.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 200").fetchall()
    
    # Get latest risk level for each patient
    latest_patient_risks = {}
    for p in preds:
        if p["patient_id"] not in latest_patient_risks:
            latest_patient_risks[p["patient_id"]] = p["risk_level"]
    
    db.close()

    total = len(patients)
    rc = {"High":0,"Medium":0,"Low":0}
    alerts = []
    
    # Count risk levels based on LATEST prediction per patient
    for risk in latest_patient_risks.values():
        if risk in rc:
            rc[risk] += 1
    
    # Get recent alerts
    for p in preds:
        if p["risk_level"]=="High" or p["fall_event"]==1:
            alerts.append(p)

    # Build alert rows
    alert_rows = "".join(f"""
      <tr class="row-alert">
        <td><strong>{a['patient_id']}</strong></td>
        <td><span class="badge bh">{a['risk_level']}</span></td>
        <td>{'🚨 Detected' if a['fall_event'] else 'None'}</td>
        <td>{a['fall_severity'] or '–'}</td>
        <td>{a['timestamp']}</td>
      </tr>""" for a in alerts[:10]) or \
        '<tr><td colspan="5" class="empty"><i class="fas fa-shield-alt"></i> No active alerts</td></tr>'

    # Build patient rows
    pt_rows = "".join(f"""
      <tr>
        <td>{p['patient_id']}</td>
        <td><strong>{p['name']}</strong></td>
        <td>{p['age']}</td>
        <td>{p['room_type']}</td>
        <td>{p['assigned_to'] or '–'}</td>
        <td>
          <a href="/sensor?pid={p['patient_id']}" class="sm-btn sm-blue">
            <i class="fas fa-microchip"></i> Monitor
          </a>
        </td>
      </tr>""" for p in patients)

    body = f"""
    <div class="page-hdr">
      <h2><i class="fas fa-stethoscope"></i> Doctor Dashboard</h2>
      <button class="btn btn-p" onclick="document.getElementById('addModal').style.display='flex'">
        <i class="fas fa-user-plus"></i> Add Patient
      </button>
    </div>

    <!-- Summary cards -->
    <div class="four-col">
      <div class="card card-blue">
        <div class="card-icon"><i class="fas fa-users"></i></div>
        <div><span class="card-num">{total}</span><span class="card-lbl">Total Patients</span></div>
      </div>
      <div class="card card-red">
        <div class="card-icon"><i class="fas fa-exclamation-triangle"></i></div>
        <div><span class="card-num">{rc['High']}</span><span class="card-lbl">High Risk</span></div>
      </div>
      <div class="card card-orange">
        <div class="card-icon"><i class="fas fa-exclamation-circle"></i></div>
        <div><span class="card-num">{rc['Medium']}</span><span class="card-lbl">Medium Risk</span></div>
      </div>
      <div class="card card-green">
        <div class="card-icon"><i class="fas fa-check-circle"></i></div>
        <div><span class="card-num">{rc['Low']}</span><span class="card-lbl">Low Risk</span></div>
      </div>
    </div>

    <!-- Charts -->
    <div class="two-col">
      <div class="panel">
        <div class="panel-hdr"><i class="fas fa-chart-pie"></i> Risk Distribution</div>
        <div class="panel-body chart-wrap"><canvas id="pieChart"></canvas></div>
      </div>
      <div class="panel">
        <div class="panel-hdr"><i class="fas fa-chart-line"></i> Recent Heart Rate Trend</div>
        <div class="panel-body chart-wrap"><canvas id="hrChart"></canvas></div>
      </div>
    </div>

    <!-- Alerts -->
    <div class="panel">
      <div class="panel-hdr panel-hdr-alert">
        <span><i class="fas fa-bell"></i> Active Alerts</span>
        <span class="badge br">{len(alerts)}</span>
      </div>
      <div class="panel-body tbl-wrap">
        <table class="tbl">
          <thead><tr>
            <th>Patient ID</th><th>Risk</th><th>Fall</th><th>Severity</th><th>Time</th>
          </tr></thead>
          <tbody>{alert_rows}</tbody>
        </table>
      </div>
    </div>

    <!-- All Patients -->
    <div class="panel">
      <div class="panel-hdr"><i class="fas fa-hospital-user"></i> All Patients</div>
      <div class="panel-body tbl-wrap">
        <table class="tbl">
          <thead><tr>
            <th>ID</th><th>Name</th><th>Age</th><th>Room</th><th>Assigned To</th><th>Action</th>
          </tr></thead>
          <tbody>{pt_rows}</tbody>
        </table>
      </div>
    </div>

    <!-- Add Patient Modal -->
    <div id="addModal" class="modal-ov" style="display:none">
      <div class="modal-card">
        <div class="modal-hdr">
          <span><i class="fas fa-user-plus"></i> Add New Patient</span>
          <button class="modal-cls" onclick="document.getElementById('addModal').style.display='none'">✕</button>
        </div>
        <div class="modal-body">
          <div id="addMsg" class="alert" style="display:none"></div>
          <div class="form-group"><label>Patient ID</label>
            <input type="text" id="apId" class="fc" placeholder="e.g. P005"></div>
          <div class="form-group"><label>Full Name</label>
            <input type="text" id="apName" class="fc" placeholder="Patient full name"></div>
          <div class="two-col-sm">
            <div class="form-group"><label>Age</label>
              <input type="number" id="apAge" class="fc" min="1" max="120" placeholder="Age"></div>
            <div class="form-group"><label>Posture</label>
              <select id="apPosture" class="fc">
                <option>Standing</option><option>Walking</option>
                <option>Sitting</option><option>Lying</option><option>Transition</option>
              </select></div>
          </div>
          <div class="form-group"><label>Assign To (username)</label>
            <input type="text" id="apAssign" class="fc" placeholder="e.g. nurse.anna"></div>
        </div>
        <div class="modal-ft">
          <button class="btn btn-s" onclick="document.getElementById('addModal').style.display='none'">Cancel</button>
          <button class="btn btn-p" onclick="submitAddPatient()">
            <i class="fas fa-save"></i> Save Patient
          </button>
        </div>
      </div>
    </div>

    <script>
    fetch('/api/analytics').then(r=>r.json()).then(data=>{{
      const rd=data.risk_dist;
      new Chart(document.getElementById('pieChart'),{{
        type:'doughnut',
        data:{{labels:['High','Medium','Low'],
          datasets:[{{data:[rd.High||0,rd.Medium||0,rd.Low||0],
            backgroundColor:['#ef4444','#f97316','#22c55e'],borderWidth:2,borderColor:'#fff'}}]}},
        options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'bottom'}}}}}}
      }});
      const hr=data.hr_trend.slice().reverse();
      new Chart(document.getElementById('hrChart'),{{
        type:'line',
        data:{{labels:hr.map((_,i)=>'#'+(i+1)),
          datasets:[{{label:'Heart Rate',data:hr.map(h=>h.heart_rate),
            borderColor:'#2563eb',backgroundColor:'rgba(37,99,235,.1)',
            tension:.4,fill:true,pointRadius:2}}]}},
        options:{{responsive:true,maintainAspectRatio:false,
          plugins:{{legend:{{display:false}}}},scales:{{y:{{beginAtZero:false}}}}}}
      }});
    }});

    async function submitAddPatient(){{
      const payload={{
        patient_id: document.getElementById('apId').value.trim(),
        name:       document.getElementById('apName').value.trim(),
        age:        document.getElementById('apAge').value,
        posture:    document.getElementById('apPosture').value,
        assigned_to:document.getElementById('apAssign').value.trim(),
      }};
      if(!payload.patient_id||!payload.name){{
        showAMsg('err','Patient ID and Name are required.');return;
      }}
      const res=await fetch('/add-patient',{{method:'POST',
        headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}});
      const d=await res.json();
      if(d.success){{showAMsg('ok','Patient added!');setTimeout(()=>location.reload(),1500)}}
      else showAMsg('err',d.error||'Failed to add patient.');
    }}
    function showAMsg(type,text){{
      const m=document.getElementById('addMsg');
      m.className=type==='ok'?'alert alert-ok':'alert alert-err';
      m.innerHTML='<i class="fas fa-'+(type==='ok'?'check-circle':'exclamation-circle')+'"></i> '+text;
      m.style.display='flex';
    }}
    </script>
    """
    return page("Doctor Dashboard", body, active="/doctor")

# ══════════════════════════════════════════════════════════════════════
# SECTION 10 – ROUTE: NURSE DASHBOARD
# ══════════════════════════════════════════════════════════════════════
@app.route("/nurse")
@login_required
def nurse():
    db = get_db()
    my_pts = db.execute(
        "SELECT p.*, u.name as doctor_name FROM patients p LEFT JOIN users u ON p.assigned_by = u.username WHERE p.assigned_to=?",
        (session["username"],)
    ).fetchall()
    alerts = db.execute("""
        SELECT p.*,pt.name as patient_name FROM predictions p
        JOIN patients pt ON p.patient_id=pt.patient_id
        WHERE (p.risk_level='High' OR p.fall_event=1) AND p.checked=0
        AND pt.assigned_to=?
        ORDER BY p.timestamp DESC LIMIT 20
    """, (session["username"],)).fetchall()
    
    # Count patients by their LATEST risk level for assigned patients
    high_risk_count = db.execute("""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT p1.patient_id FROM predictions p1 
            JOIN patients pt ON p1.patient_id=pt.patient_id
            WHERE p1.risk_level='High' AND pt.assigned_to=?
            AND p1.timestamp = (SELECT MAX(timestamp) FROM predictions p2 WHERE p2.patient_id = p1.patient_id)
        )
    """, (session["username"],)).fetchone()[0]
    
    medium_risk_count = db.execute("""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT p1.patient_id FROM predictions p1 
            JOIN patients pt ON p1.patient_id=pt.patient_id
            WHERE p1.risk_level='Medium' AND pt.assigned_to=?
            AND p1.timestamp = (SELECT MAX(timestamp) FROM predictions p2 WHERE p2.patient_id = p1.patient_id)
        )
    """, (session["username"],)).fetchone()[0]
    
    low_risk_count = db.execute("""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT p1.patient_id FROM predictions p1 
            JOIN patients pt ON p1.patient_id=pt.patient_id
            WHERE p1.risk_level='Low' AND pt.assigned_to=?
            AND p1.timestamp = (SELECT MAX(timestamp) FROM predictions p2 WHERE p2.patient_id = p1.patient_id)
        )
    """, (session["username"],)).fetchone()[0]
    
    db.close()

    al_html = "".join(f"""
      <div class="al-card" id="al-{a['id']}">
        <div class="al-left">
          <div class="al-icon"><i class="fas fa-exclamation-triangle"></i></div>
          <div>
            <strong>{a['patient_name']}</strong> ({a['patient_id']})<br>
            <span class="badge bh">{a['risk_level']} Risk</span>
            {'<span class="badge br">Fall Detected</span>' if a['fall_event'] else ''}
            <span class="badge bgr">{a['fall_severity'] or ''}</span>
          </div>
        </div>
        <div class="al-right">
          <small>{a['timestamp']}</small>
          <button class="sm-btn sm-green" onclick="markChecked({a['id']})">
            <i class="fas fa-check"></i> Mark Checked
          </button>
        </div>
      </div>""" for a in alerts) or \
      '<p class="empty"><i class="fas fa-shield-alt"></i> No active alerts – all patients stable</p>'

    pt_html = "".join(f"""
      <div class="pt-card">
        <div class="pt-avatar"><i class="fas fa-user"></i></div>
        <div class="pt-info">
          <h4>{p['name']}</h4>
          <p>ID: <strong>{p['patient_id']}</strong></p>
          <p>Age: {p['age']} &nbsp;|&nbsp; Room: {p['room_type']}</p>
          <p>Assigned by: <strong>{p['doctor_name'] or 'Unknown'}</strong></p>
        </div>
        <div class="pt-actions">
          <a href="/sensor?pid={p['patient_id']}" class="sm-btn sm-blue">
            <i class="fas fa-microchip"></i> Monitor
          </a>
          <a href="/history?pid={p['patient_id']}" class="sm-btn sm-grey">
            <i class="fas fa-history"></i> History
          </a>
        </div>
      </div>""" for p in my_pts) or \
      '<p class="empty">No patients assigned yet.</p>'

    body = f"""
    <div class="page-hdr">
      <h2><i class="fas fa-user-nurse"></i> Nurse Dashboard</h2>
      <a href="/sensor" class="btn btn-p">
        <i class="fas fa-plus-circle"></i> Enter Sensor Data
      </a>
    </div>

    <!-- Risk Distribution Cards -->
    <div class="three-col">
      <div class="metric-card">
        <div class="metric-icon" style="background:#ef4444"><i class="fas fa-exclamation-circle"></i></div>
        <div class="metric-info">
          <div class="metric-label">High Risk Patients</div>
          <div class="metric-value">{high_risk_count}</div>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-icon" style="background:#f97316"><i class="fas fa-exclamation"></i></div>
        <div class="metric-info">
          <div class="metric-label">Medium Risk Patients</div>
          <div class="metric-value">{medium_risk_count}</div>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-icon bg-green"><i class="fas fa-check-circle"></i></div>
        <div class="metric-info">
          <div class="metric-label">Low Risk Patients</div>
          <div class="metric-value">{low_risk_count}</div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-hdr panel-hdr-alert">
        <span><i class="fas fa-bell"></i> 🚨 Active Alerts</span>
        <span class="badge br">{len(alerts)}</span>
      </div>
      <div class="panel-body">{al_html}</div>
    </div>

    <div class="panel">
      <div class="panel-hdr"><i class="fas fa-hospital-user"></i> My Assigned Patients</div>
      <div class="panel-body"><div class="pt-grid">{pt_html}</div></div>
    </div>

    <div class="panel">
      <div class="panel-hdr"><i class="fas fa-clipboard-list"></i> Care Recommendations</div>
      <div class="panel-body">
        <div class="recs">
          <div class="rec rec-r"><i class="fas fa-exclamation-triangle"></i>
            <div><strong>High Risk Patients</strong><br>
            Provide immediate physical assistance. Apply non-slip footwear. Raise bed rails. Alert doctor immediately.</div></div>
          <div class="rec rec-o"><i class="fas fa-eye"></i>
            <div><strong>Medium Risk Patients</strong><br>
            Monitor every 30 minutes. Ensure clear pathways. Check mobility aids. Review medications for dizziness.</div></div>
          <div class="rec rec-g"><i class="fas fa-check-circle"></i>
            <div><strong>Low Risk Patients</strong><br>
            Standard hourly rounds. Encourage call-bell use. Ensure adequate lighting in rooms and bathrooms.</div></div>
        </div>
      </div>
    </div>

    <script>
    async function markChecked(id){{
      await fetch('/mark-checked/'+id,{{method:'POST'}});
      const el=document.getElementById('al-'+id);
      if(el){{el.style.opacity='.4';el.style.pointerEvents='none'}}
      showToast('Alert marked as checked.','ok');
    }}
    </script>
    """
    return page("Nurse Dashboard", body, active="/nurse")

# ══════════════════════════════════════════════════════════════════════
# SECTION 11 – ROUTE: SENSOR INPUT & PREDICTION
# ══════════════════════════════════════════════════════════════════════
@app.route("/sensor")
@login_required
def sensor():
    db = get_db()
    patients = db.execute("SELECT * FROM patients").fetchall()
    db.close()
    pid = request.args.get("pid","")

    pt_opts = "".join(
        f'<option value="{p["patient_id"]}" {"selected" if p["patient_id"]==pid else ""}>'
        f'{p["patient_id"]} – {p["name"]}</option>'
        for p in patients
    )

    body = f"""
    <div class="page-hdr">
      <h2><i class="fas fa-microchip"></i> Sensor Data Input</h2>
      <p class="subtitle">Enter patient sensor readings to predict fall risk</p>
    </div>

    <div class="sensor-wrap">
      <!-- Input Form -->
      <div class="panel">
        <div class="panel-hdr"><i class="fas fa-keyboard"></i> Enter Readings</div>
        <div class="panel-body">
          <div class="form-group">
            <label><i class="fas fa-hospital-user"></i> Select Patient <span class="req">*</span></label>
            <select id="patientId" class="fc">
              <option value="">-- Select a patient --</option>
              {pt_opts}
            </select>
          </div>

          <div class="sensor-lbl">📡 Accelerometer (g)</div>
          <div class="three-col">
            <div class="form-group"><label>acc_x</label>
              <input type="number" id="acc_x" class="fc" step="0.001" value="0.5"></div>
            <div class="form-group"><label>acc_y</label>
              <input type="number" id="acc_y" class="fc" step="0.001" value="1.4"></div>
            <div class="form-group"><label>acc_z</label>
              <input type="number" id="acc_z" class="fc" step="0.001" value="9.1"></div>
          </div>

          <div class="sensor-lbl">🔄 Gyroscope (°/s)</div>
          <div class="three-col">
            <div class="form-group"><label>gyro_x</label>
              <input type="number" id="gyro_x" class="fc" step="0.001" value="0.1"></div>
            <div class="form-group"><label>gyro_y</label>
              <input type="number" id="gyro_y" class="fc" step="0.001" value="0.2"></div>
            <div class="form-group"><label>gyro_z</label>
              <input type="number" id="gyro_z" class="fc" step="0.001" value="0.05"></div>
          </div>

          <div class="sensor-lbl">🏥 Environment</div>
          <div class="three-col">
            <div class="form-group"><label>Heart Rate (bpm)</label>
              <input type="number" id="heart_rate" class="fc" value="75" min="30" max="200"></div>
            <div class="form-group"><label>Room Temp (°C)</label>
              <input type="number" id="room_temp" class="fc" value="22"></div>
            <div class="form-group"><label>Room Light (lux)</label>
              <input type="number" id="room_light" class="fc" value="300"></div>
          </div>

          <div class="three-col">
            <div class="form-group"><label>Room Type</label>
              <select id="room_type" class="fc">
                <option>Bedroom</option><option>Bathroom</option>
                <option>LivingRoom</option><option>Hallway</option>
              </select></div>
            <div class="form-group"><label>Posture</label>
              <select id="posture" class="fc">
                <option>Standing</option><option>Walking</option>
                <option>Sitting</option><option>Lying</option><option>Transition</option>
              </select></div>
            <div class="form-group"><label>Time of Day</label>
              <select id="time_of_day" class="fc">
                <option>Morning</option><option>Afternoon</option>
                <option>Evening</option><option>Night</option>
              </select></div>
          </div>

          <div class="btn-row">
            <button class="btn btn-s" onclick="fillRandom()">
              <i class="fas fa-random"></i> Random Sample
            </button>
            <button class="btn btn-s" onclick="fillHigh()">
              <i class="fas fa-exclamation-triangle"></i> High Risk Sample
            </button>
          </div>
          <button id="predictBtn" class="btn btn-p btn-full" onclick="submitSensor()">
            <i class="fas fa-brain"></i> Predict Risk
          </button>
        </div>
      </div>

      <!-- Result Panel -->
      <div id="resultPanel" style="display:none">
        <div class="panel">
          <div class="panel-hdr"><i class="fas fa-chart-bar"></i> Prediction Result</div>
          <div class="panel-body" id="resultBody"></div>
        </div>
      </div>
    </div>

    <!-- Fall Alert Modal -->
    <div id="fallModal" class="modal-ov" style="display:none">
      <div class="modal-card">
        <div class="modal-hdr modal-hdr-alert">
          <span><i class="fas fa-bell"></i> 🚨 Emergency Alert</span>
        </div>
        <div class="modal-body text-center">
          <div class="alert-pulse">⚠️</div>
          <h3 id="alertTitle">Fall Detected!</h3>
          <p id="alertDesc" style="margin-top:10px;color:#555"></p>
        </div>
        <div class="modal-ft" style="justify-content:center">
          <button class="btn btn-d btn-full"
            onclick="document.getElementById('fallModal').style.display='none'">
            <i class="fas fa-check"></i> Acknowledge Alert
          </button>
        </div>
      </div>
    </div>

    <script>
    function v(id){{return document.getElementById(id).value}}
    function set(id,val){{document.getElementById(id).value=val}}

    function fillRandom(){{
      set('acc_x',(Math.random()*4-2).toFixed(3));
      set('acc_y',(Math.random()*4-2).toFixed(3));
      set('acc_z',(8+Math.random()*3).toFixed(3));
      set('gyro_x',(Math.random()*2-1).toFixed(3));
      set('gyro_y',(Math.random()*2-1).toFixed(3));
      set('gyro_z',(Math.random()*2-1).toFixed(3));
      set('heart_rate',60+Math.floor(Math.random()*50));
    }}
    function fillHigh(){{
      set('acc_x',2.8);set('acc_y',-3.1);set('acc_z',4.2);
      set('gyro_x',3.5);set('gyro_y',-2.9);set('gyro_z',2.1);
      set('heart_rate',118);set('room_temp',28);set('room_light',50);
      document.getElementById('room_type').value='Bathroom';
      document.getElementById('posture').value='Transition';
      document.getElementById('time_of_day').value='Night';
    }}

    async function submitSensor(){{
      const pid=v('patientId');
      if(!pid){{showToast('Please select a patient.','err');return;}}
      const btn=document.getElementById('predictBtn');
      btn.disabled=true;btn.innerHTML='<i class="fas fa-spinner fa-spin"></i> Analysing…';
      const payload={{
        patient_id:pid,
        acc_x:+v('acc_x'),acc_y:+v('acc_y'),acc_z:+v('acc_z'),
        gyro_x:+v('gyro_x'),gyro_y:+v('gyro_y'),gyro_z:+v('gyro_z'),
        heart_rate:+v('heart_rate'),room_temp:+v('room_temp'),room_light:+v('room_light'),
        room_type:v('room_type'),posture:v('posture'),time_of_day:v('time_of_day'),
      }};
      try{{
        const res=await fetch('/predict',{{method:'POST',
          headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}});
        const p=await res.json();
        showResult(p);
        if(p.risk_level==='High'||p.fall_event===1)showAlert(p);
      }}finally{{
        btn.disabled=false;btn.innerHTML='<i class="fas fa-brain"></i> Predict Risk';
      }}
    }}

    function riskCls(r){{return r==='High'?'risk-r':r==='Medium'?'risk-o':'risk-g'}}
    function recCls(r){{return r==='High'?'rec-r':r==='Medium'?'rec-o':'rec-g'}}
    function recText(r){{
      return r==='High'?'🚨 Immediate assistance required. Alert the medical team now!'
        :r==='Medium'?'👁️ Monitor closely every 30 minutes. Check environment.'
        :'✅ Standard care. Continue normal monitoring rounds.';
    }}

    function showResult(p){{
      const panel=document.getElementById('resultPanel');
      const body=document.getElementById('resultBody');
      body.innerHTML=`
        <div class="res-patient">
          <i class="fas fa-user-circle"></i>
          <div><strong>${{p.patient_name}}</strong>
            <span class="muted">(${{p.patient_id}})</span></div>
        </div>
        <div class="risk-card ${{riskCls(p.risk_level)}}">
          <div class="risk-lbl">Risk Level</div>
          <div class="risk-val">${{p.risk_level}}</div>
          <div class="risk-prob">Confidence: ${{p.risk_prob}}%</div>
          <div class="prob-bar"><div class="prob-fill" style="width:${{p.risk_prob}}%"></div></div>
        </div>
        <div class="res-badges">
          <div class="res-badge ${{p.fall_event?'br bg':'bg bg'}} badge">
            <i class="fas fa-${{p.fall_event?'exclamation-triangle':'check-circle'}}"></i>
            ${{p.fall_event?'Fall Detected':'No Fall Detected'}}
          </div>
          <div class="res-badge bgr badge">
            <i class="fas fa-layer-group"></i> Severity: ${{p.fall_severity}}
          </div>
        </div>
        <div class="explain">
          <strong><i class="fas fa-lightbulb"></i> Explanation</strong>
          <p>${{p.explanation}}</p>
        </div>
        <div class="recomm ${{recCls(p.risk_level)}}">
          <strong><i class="fas fa-clipboard-list"></i> Recommendation</strong>
          <p>${{recText(p.risk_level)}}</p>
        </div>`;
      panel.style.display='block';
      panel.scrollIntoView({{behavior:'smooth',block:'start'}});
    }}

    function showAlert(p){{
      document.getElementById('alertTitle').textContent=
        p.fall_event?'🚨 Emergency: Fall Detected!':'⚠️ High Fall Risk!';
      document.getElementById('alertDesc').textContent=
        'Patient '+p.patient_name+' ('+p.patient_id+') – '+p.risk_level+' Risk. '+p.explanation;
      document.getElementById('fallModal').style.display='flex';
    }}
    </script>
    """
    return page("Sensor Input", body, active="/sensor")

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    d = request.json
    result = run_prediction({k: float(d[k]) if k not in
        ('room_type','posture','time_of_day','patient_id') else d[k]
        for k in d})
    db = get_db()
    patient = db.execute("SELECT * FROM patients WHERE patient_id=?", (d["patient_id"],)).fetchone()
    db.execute("""INSERT INTO predictions
        (patient_id,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,heart_rate,
         room_temp,room_light,room_type,posture,time_of_day,
         risk_level,fall_event,fall_severity,risk_prob,timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (d["patient_id"], d["acc_x"], d["acc_y"], d["acc_z"],
         d["gyro_x"], d["gyro_y"], d["gyro_z"], d["heart_rate"],
         d["room_temp"], d["room_light"], d["room_type"], d["posture"],
         d["time_of_day"], result["risk_level"], result["fall_event"],
         result["fall_severity"], result["risk_prob"],
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    db.commit(); db.close()
    result["patient_name"] = patient["name"] if patient else "Unknown"
    result["patient_id"]   = d["patient_id"]
    return jsonify(result)

# ══════════════════════════════════════════════════════════════════════
# SECTION 12 – ROUTE: PATIENT HISTORY
# ══════════════════════════════════════════════════════════════════════
@app.route("/history")
@login_required
def history():
    db = get_db()
    patients = db.execute("SELECT * FROM patients").fetchall()
    db.close()
    pid = request.args.get("pid","")

    pt_opts = '<option value="">All Patients</option>' + "".join(
        f'<option value="{p["patient_id"]}" {"selected" if p["patient_id"]==pid else ""}>'
        f'{p["patient_id"]} – {p["name"]}</option>'
        for p in patients
    )

    body = f"""
    <div class="page-hdr">
      <h2><i class="fas fa-history"></i> Prediction History</h2>
    </div>

    <div class="panel">
      <div class="panel-body">
        <div class="filter-row">
          <div class="form-group"><label>Search</label>
            <input type="text" id="srch" class="fc" placeholder="Name or ID…" oninput="load()"></div>
          <div class="form-group"><label>Patient</label>
            <select id="ptF" class="fc" onchange="load()">{pt_opts}</select></div>
          <div class="form-group"><label>Risk Level</label>
            <select id="riskF" class="fc" onchange="load()">
              <option value="">All</option>
              <option>High</option><option>Medium</option><option>Low</option>
            </select></div>
          <div class="form-group"><label>Sort By</label>
            <select id="sortF" class="fc" onchange="load()">
              <option value="timestamp DESC">Newest First</option>
              <option value="timestamp ASC">Oldest First</option>
            </select></div>
          <div class="form-group align-end">
            <button class="btn btn-s" onclick="load()"><i class="fas fa-search"></i> Search</button>
            <button class="btn btn-s" onclick="clearF()"><i class="fas fa-times"></i> Clear</button>
          </div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-hdr">
        <span><i class="fas fa-table"></i> Records</span>
        <span id="cnt" class="badge bb">Loading…</span>
      </div>
      <div class="panel-body tbl-wrap">
        <table class="tbl">
          <thead><tr>
            <th>#</th><th>Patient</th><th>Risk</th><th>Fall</th><th>Severity</th>
            <th>HR</th><th>Posture</th><th>Room</th><th>Time</th><th>Timestamp</th>
          </tr></thead>
          <tbody id="hBody">
            <tr><td colspan="10" class="loading-row">
              <i class="fas fa-spinner fa-spin"></i> Loading…</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <script>
    async function load(){{
      const p=new URLSearchParams({{
        search:document.getElementById('srch').value,
        patient_id:document.getElementById('ptF').value,
        risk:document.getElementById('riskF').value,
        sort:document.getElementById('sortF').value
      }});
      const rows=await fetch('/api/history?'+p).then(r=>r.json());
      document.getElementById('cnt').textContent=rows.length+' records';
      const tbody=document.getElementById('hBody');
      if(!rows.length){{
        tbody.innerHTML='<tr><td colspan="10" class="empty">No records found.</td></tr>';return;
      }}
      const bc=r=>r==='High'?'bh':r==='Medium'?'bm':'bl';
      tbody.innerHTML=rows.map((r,i)=>`
        <tr class="${{r.risk_level==='High'?'row-alert':''}}">
          <td>${{i+1}}</td>
          <td><strong>${{r.patient_name||r.patient_id}}</strong><br>
            <small class="muted">${{r.patient_id}}</small></td>
          <td><span class="badge ${{bc(r.risk_level)}}">${{r.risk_level}}</span></td>
          <td>${{r.fall_event
            ?'<span class="badge br">Detected</span>'
            :'<span class="badge bl">None</span>'}}</td>
          <td>${{r.fall_severity||'–'}}</td>
          <td>${{r.heart_rate}} bpm</td>
          <td>${{r.posture}}</td>
          <td>${{r.room_type}}</td>
          <td>${{r.time_of_day}}</td>
          <td><small>${{r.timestamp}}</small></td>
        </tr>`).join('');
    }}
    function clearF(){{
      ['srch','ptF','riskF'].forEach(x=>document.getElementById(x).value='');
      document.getElementById('sortF').value='timestamp DESC';load();
    }}
    window.addEventListener('load', load);
    </script>
    """
    return page("History", body, active="/history")

# ══════════════════════════════════════════════════════════════════════
# SECTION 13 – ROUTE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════
@app.route("/analytics")
@login_required
@doctor_required
def analytics():
    body = """
    <div class="page-hdr">
      <h2><i class="fas fa-chart-bar"></i> Analytics &amp; Statistics</h2>
    </div>

    <div class="four-col">
      <div class="card card-blue"><div class="card-icon"><i class="fas fa-database"></i></div>
        <div><span class="card-num" id="sTot">–</span><span class="card-lbl">Total Predictions</span></div></div>
      <div class="card card-red"><div class="card-icon"><i class="fas fa-exclamation-triangle"></i></div>
        <div><span class="card-num" id="sHigh">–</span><span class="card-lbl">High Risk Events</span></div></div>
      <div class="card card-orange"><div class="card-icon"><i class="fas fa-person-falling"></i></div>
        <div><span class="card-num" id="sFalls">–</span><span class="card-lbl">Falls Detected</span></div></div>
      <div class="card card-green"><div class="card-icon"><i class="fas fa-heart-pulse"></i></div>
        <div><span class="card-num" id="sHR">–</span><span class="card-lbl">Avg Heart Rate</span></div></div>
    </div>

    <div class="two-col">
      <div class="panel">
        <div class="panel-hdr"><i class="fas fa-chart-pie"></i> Risk Distribution</div>
        <div class="panel-body chart-wrap"><canvas id="pieChart"></canvas></div>
      </div>
      <div class="panel">
        <div class="panel-hdr"><i class="fas fa-chart-bar"></i> Risk Level Counts</div>
        <div class="panel-body chart-wrap"><canvas id="barChart"></canvas></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-hdr"><i class="fas fa-chart-line"></i> Heart Rate Trend (last 50 readings)</div>
      <div class="panel-body" style="height:220px"><canvas id="lineChart"></canvas></div>
    </div>

    <div class="panel">
      <div class="panel-hdr"><i class="fas fa-table"></i> Per-Patient Risk Summary</div>
      <div class="panel-body tbl-wrap">
        <table class="tbl">
          <thead><tr><th>Patient ID</th><th>Name</th><th>Last Risk</th><th>Last Seen</th></tr></thead>
          <tbody id="sumBody"><tr><td colspan="4" class="loading-row">
            <i class="fas fa-spinner fa-spin"></i></td></tr></tbody>
        </table>
      </div>
    </div>

    <script>
    fetch('/api/analytics').then(r=>r.json()).then(data=>{
      const rd=data.risk_dist;
      const total=(rd.High||0)+(rd.Medium||0)+(rd.Low||0);
      document.getElementById('sTot').textContent=total;
      document.getElementById('sHigh').textContent=rd.High||0;

      const falls=data.hr_trend.filter(h=>h.heart_rate>110).length;
      document.getElementById('sFalls').textContent=falls;
      const avgHR=data.hr_trend.length
        ?Math.round(data.hr_trend.reduce((s,h)=>s+h.heart_rate,0)/data.hr_trend.length):0;
      document.getElementById('sHR').textContent=avgHR+' bpm';

      new Chart(document.getElementById('pieChart'),{
        type:'doughnut',
        data:{labels:['High','Medium','Low'],
          datasets:[{data:[rd.High||0,rd.Medium||0,rd.Low||0],
            backgroundColor:['#ef4444','#f97316','#22c55e'],borderWidth:2,borderColor:'#fff'}]},
        options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom'}}}
      });
      new Chart(document.getElementById('barChart'),{
        type:'bar',
        data:{labels:['High Risk','Medium Risk','Low Risk'],
          datasets:[{label:'Predictions',
            data:[rd.High||0,rd.Medium||0,rd.Low||0],
            backgroundColor:['rgba(239,68,68,.8)','rgba(249,115,22,.8)','rgba(34,197,94,.8)'],
            borderRadius:6}]},
        options:{responsive:true,maintainAspectRatio:false,
          plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}}
      });

      const trend=data.hr_trend.slice().reverse();
      new Chart(document.getElementById('lineChart'),{
        type:'line',
        data:{labels:trend.map((_,i)=>'#'+(i+1)),
          datasets:[{label:'Heart Rate',data:trend.map(h=>h.heart_rate),
            borderColor:'#2563eb',backgroundColor:'rgba(37,99,235,.1)',
            tension:.4,fill:true,pointRadius:2}]},
        options:{responsive:true,maintainAspectRatio:false,
          plugins:{legend:{display:false}},scales:{y:{beginAtZero:false}}}
      });

      const seen={};
      data.patient_risk.forEach(r=>{if(!seen[r.patient_id])seen[r.patient_id]=r;});
      const bc=r=>r==='High'?'bh':r==='Medium'?'bm':'bl';
      document.getElementById('sumBody').innerHTML=Object.values(seen).map(r=>`
        <tr>
          <td>${r.patient_id}</td>
          <td><strong>${r.name}</strong></td>
          <td><span class="badge ${bc(r.risk_level)}">${r.risk_level}</span></td>
          <td><small>${r.timestamp}</small></td>
        </tr>`).join('');
    });
    </script>
    """
    return page("Analytics", body, active="/analytics")

# ══════════════════════════════════════════════════════════════════════
# SECTION 14 – API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════
@app.route("/add-patient", methods=["POST"])
@login_required
@doctor_required
def add_patient():
    d = request.json
    db = get_db()
    try:
        db.execute("""INSERT INTO patients
            (patient_id,name,age,room_type,assigned_to,assigned_by,created_at) VALUES (?,?,?,?,?,?,?)""",
            (d["patient_id"], d["name"], d.get("age"), d.get("posture","Standing"),
             d.get("assigned_to",""), session["username"], datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        db.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    finally:
        db.close()

@app.route("/mark-checked/<int:pid>", methods=["POST"])
@login_required
def mark_checked(pid):
    db = get_db()
    db.execute("UPDATE predictions SET checked=1 WHERE id=?", (pid,))
    db.commit(); db.close()
    return jsonify({"success": True})

@app.route("/api/history")
@login_required
def api_history():
    db = get_db()
    q = """SELECT p.*,pt.name as patient_name FROM predictions p
           LEFT JOIN patients pt ON p.patient_id=pt.patient_id WHERE 1=1"""
    params = []
    s = request.args.get("search","")
    pid = request.args.get("patient_id","")
    risk = request.args.get("risk","")
    sort = request.args.get("sort","timestamp DESC")
    
    # Validate sort parameter for security
    allowed_sorts = {
        'timestamp DESC': 'p.timestamp DESC',
        'timestamp ASC': 'p.timestamp ASC'
    }
    sort_clause = allowed_sorts.get(sort, 'p.timestamp DESC')
    
    if s:
        q += " AND (p.patient_id LIKE ? OR pt.name LIKE ?)"; params += [f"%{s}%"]*2
    if pid:
        q += " AND p.patient_id=?"; params.append(pid)
    if risk:
        q += " AND p.risk_level=?"; params.append(risk)
    q += f" ORDER BY {sort_clause} LIMIT 100"
    rows = db.execute(q, params).fetchall(); db.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/analytics")
@login_required
def api_analytics():
    db = get_db()
    rd = dict(db.execute(
        "SELECT risk_level,COUNT(*) FROM predictions GROUP BY risk_level"
    ).fetchall())
    hr = db.execute(
        "SELECT patient_id,heart_rate,timestamp FROM predictions ORDER BY timestamp DESC LIMIT 50"
    ).fetchall()
    pr = db.execute("""
        SELECT p.patient_id,pt.name,p.risk_level,p.timestamp
        FROM predictions p JOIN patients pt ON p.patient_id=pt.patient_id
        ORDER BY p.timestamp DESC LIMIT 100
    """).fetchall()
    db.close()
    return jsonify(dict(risk_dist=rd,
                        hr_trend=[dict(r) for r in hr],
                        patient_risk=[dict(r) for r in pr]))

# ══════════════════════════════════════════════════════════════════════
# SECTION 15 – ADMIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════
@app.route("/admin")
@login_required
@admin_required
def admin_dashboard():
    """Admin dashboard displaying system overview and user management."""
    db = get_db()
    
    # Get metrics
    total_patients = db.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    total_doctors = db.execute("SELECT COUNT(*) FROM users WHERE role='Doctor'").fetchone()[0]
    total_nurses = db.execute("SELECT COUNT(*) FROM users WHERE role='Nurse'").fetchone()[0]
    total_predictions = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    
    # Count patients by their LATEST risk level (current status)
    high_risk_count = db.execute("""
        SELECT COUNT(*) FROM (
            SELECT patient_id FROM predictions p1 
            WHERE risk_level='High' 
            AND timestamp = (SELECT MAX(timestamp) FROM predictions p2 WHERE p2.patient_id = p1.patient_id)
        )
    """).fetchone()[0]
    
    medium_risk_count = db.execute("""
        SELECT COUNT(*) FROM (
            SELECT patient_id FROM predictions p1 
            WHERE risk_level='Medium' 
            AND timestamp = (SELECT MAX(timestamp) FROM predictions p2 WHERE p2.patient_id = p1.patient_id)
        )
    """).fetchone()[0]
    
    low_risk_count = db.execute("""
        SELECT COUNT(*) FROM (
            SELECT patient_id FROM predictions p1 
            WHERE risk_level='Low' 
            AND timestamp = (SELECT MAX(timestamp) FROM predictions p2 WHERE p2.patient_id = p1.patient_id)
        )
    """).fetchone()[0]
    
    # Get users list
    users = db.execute("SELECT id,username,name,role,status,created_at FROM users ORDER BY created_at DESC").fetchall()
    
    # Get patients list
    patients = db.execute("""
        SELECT DISTINCT p.patient_id,p.name,p.room_type,
        (SELECT risk_level FROM predictions WHERE patient_id=p.patient_id ORDER BY timestamp DESC LIMIT 1) as risk_level
        FROM patients p ORDER BY p.patient_id DESC LIMIT 50
    """).fetchall()
    
    # Get recent alerts
    alerts = db.execute("""
        SELECT patient_id,risk_level,fall_event,fall_severity,timestamp 
        FROM predictions WHERE risk_level='High' OR fall_event=1 
        ORDER BY timestamp DESC LIMIT 20
    """).fetchall()
    
    db.close()
    
    # Build users table
    users_rows = "".join(f"""
      <tr>
        <td>{u['name']}</td>
        <td><span class="badge {'bd' if u['role']=='Doctor' else 'bn'}">{u['role']}</span></td>
        <td><span class="badge {'bs-ok' if u['status']=='Active' else 'bs-warn'}">{u['status']}</span></td>
        <td class="action-cell">
          <button class="sm-btn sm-blue" onclick="editUser({u['id']},'{u['username']}','{u['name']}','{u['role']}')">
            <i class="fas fa-edit"></i>
          </button>
          <button class="sm-btn sm-red" onclick="deleteUser({u['id']})">
            <i class="fas fa-trash"></i>
          </button>
        </td>
      </tr>""" for u in users)
    
    # Build patients table
    patients_rows = "".join(f"""
      <tr>
        <td><strong>{p['patient_id']}</strong></td>
        <td>{p['name']}</td>
        <td>{p['room_type']}</td>
        <td><span class="badge {'br' if p['risk_level']=='High' else 'by' if p['risk_level']=='Medium' else 'bg'}">{p['risk_level'] or 'N/A'}</span></td>
      </tr>""" for p in patients)
    
    # Build alerts table
    alerts_rows = "".join(f"""
      <tr class="{'row-alert' if a['risk_level']=='High' or a['fall_event'] else ''}">
        <td><strong>{a['patient_id']}</strong></td>
        <td><span class="badge {'br' if a['risk_level']=='High' else 'by' if a['risk_level']=='Medium' else 'bg'}">{a['risk_level']}</span></td>
        <td>{'🚨 DETECTED' if a['fall_event'] else 'None'}</td>
        <td>{a['fall_severity'] or '–'}</td>
        <td>{a['timestamp']}</td>
      </tr>""" for a in alerts) or "<tr><td colspan='5' class='empty'>No alerts</td></tr>"
    
    body = f"""
    <div class="page-hdr">
      <h2><i class="fas fa-crown"></i> System Administrator Dashboard</h2>
      <button class="btn btn-p" onclick="document.getElementById('userModal').style.display='flex'">
        <i class="fas fa-user-plus"></i> Add User
      </button>
    </div>

    <!-- System Metrics Cards -->
    <div class="three-col">
      <div class="metric-card">
        <div class="metric-icon bg-blue"><i class="fas fa-users"></i></div>
        <div class="metric-info">
          <div class="metric-label">Total Patients</div>
          <div class="metric-value">{total_patients}</div>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-icon bg-green"><i class="fas fa-stethoscope"></i></div>
        <div class="metric-info">
          <div class="metric-label">Total Doctors</div>
          <div class="metric-value">{total_doctors}</div>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-icon bg-purple"><i class="fas fa-heart"></i></div>
        <div class="metric-info">
          <div class="metric-label">Total Nurses</div>
          <div class="metric-value">{total_nurses}</div>
        </div>
      </div>
    </div>

    <!-- Risk Distribution Cards -->
    <div class="three-col">
      <div class="metric-card">
        <div class="metric-icon" style="background:#ef4444"><i class="fas fa-exclamation-circle"></i></div>
        <div class="metric-info">
          <div class="metric-label">High Risk Patients</div>
          <div class="metric-value">{high_risk_count}</div>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-icon" style="background:#f97316"><i class="fas fa-exclamation"></i></div>
        <div class="metric-info">
          <div class="metric-label">Medium Risk Patients</div>
          <div class="metric-value">{medium_risk_count}</div>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-icon" style="background:#22c55e"><i class="fas fa-check-circle"></i></div>
        <div class="metric-info">
          <div class="metric-label">Low Risk Patients</div>
          <div class="metric-value">{low_risk_count}</div>
        </div>
      </div>
    </div>

    <!-- User Management Section -->
    <div class="panel">
      <div class="panel-hdr"><i class="fas fa-users-cog"></i> User Management</div>
      <div class="panel-body tbl-wrap">
        <table class="tbl">
          <thead>
            <tr>
              <th>Name</th>
              <th>Role</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {users_rows or '<tr><td colspan="4" class="empty">No users</td></tr>'}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Patient Overview Section -->
    <div class="two-col">
      <div class="panel">
        <div class="panel-hdr"><i class="fas fa-hospital-user"></i> Patients Overview</div>
        <div class="panel-body tbl-wrap">
          <table class="tbl">
            <thead>
              <tr>
                <th>Patient ID</th>
                <th>Name</th>
                <th>Room</th>
                <th>Risk Level</th>
              </tr>
            </thead>
            <tbody>
              {patients_rows or '<tr><td colspan="4" class="empty">No patients</td></tr>'}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Alerts & Activity Section -->
      <div class="panel">
        <div class="panel-hdr"><i class="fas fa-bell"></i> Recent Alerts & Events</div>
        <div class="panel-body tbl-wrap">
          <table class="tbl">
            <thead>
              <tr>
                <th>Patient ID</th>
                <th>Risk Level</th>
                <th>Fall Detected</th>
                <th>Severity</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {alerts_rows}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Add/Edit User Modal -->
    <div id="userModal" class="modal-ov" style="display:none">
      <div class="modal-card">
        <div class="modal-hdr">
          <span><i class="fas fa-user-plus"></i> <span id="modalTitle">Add New User</span></span>
          <button class="modal-cls" onclick="document.getElementById('userModal').style.display='none'">✕</button>
        </div>
        <div class="modal-body">
          <div id="userMsg" class="alert" style="display:none"></div>
          <input type="hidden" id="userId">
          <div class="form-group">
            <label>Full Name</label>
            <input type="text" id="userName" class="fc" placeholder="Doctor/Nurse full name">
          </div>
          <div class="form-group">
            <label>Username</label>
            <input type="text" id="userUsername" class="fc" placeholder="e.g. dr.smith or nurse.anna">
          </div>
          <div class="form-group">
            <label>Password</label>
            <input type="password" id="userPassword" class="fc" placeholder="Enter password">
          </div>
          <div class="form-group">
            <label>Role</label>
            <select id="userRole" class="fc">
              <option>Doctor</option>
              <option>Nurse</option>
              <option>Admin</option>
            </select>
          </div>
          <div class="form-group">
            <label>Status</label>
            <select id="userStatus" class="fc">
              <option>Active</option>
              <option>Inactive</option>
            </select>
          </div>
        </div>
        <div class="modal-ft">
          <button class="btn btn-s" onclick="document.getElementById('userModal').style.display='none'">Cancel</button>
          <button class="btn btn-p" onclick="saveUser()">
            <i class="fas fa-save"></i> Save User
          </button>
        </div>
      </div>
    </div>

    <!-- Analytics Section -->
    <div class="three-col">
      <div class="panel">
        <div class="panel-hdr"><i class="fas fa-chart-pie"></i> Risk Distribution</div>
        <div class="panel-body" style="height:250px"><canvas id="adminPieChart"></canvas></div>
      </div>
      <div class="panel">
        <div class="panel-hdr"><i class="fas fa-chart-bar"></i> Predictions Trend</div>
        <div class="panel-body" style="height:250px"><canvas id="adminBarChart"></canvas></div>
      </div>
      <div class="panel">
        <div class="panel-hdr"><i class="fas fa-chart-line"></i> System Activity</div>
        <div class="panel-body" style="height:250px"><canvas id="adminLineChart"></canvas></div>
      </div>
    </div>

    <!-- Export Report Section -->
    <div class="panel">
      <div class="panel-hdr"><i class="fas fa-file-download"></i> System Reports</div>
      <div class="panel-body" style="display:flex; gap:10px; flex-wrap:wrap">
        <button class="btn btn-p" onclick="downloadReportCSV()">
          <i class="fas fa-download"></i> Download CSV Report
        </button>
        <button class="btn btn-p" onclick="downloadUserList()">
          <i class="fas fa-user-tie"></i> Download User List
        </button>
        <button class="btn btn-p" onclick="downloadAlertLog()">
          <i class="fas fa-bell"></i> Download Alert Log
        </button>
      </div>
    </div>

    <script>
    // Charts initialization
    function showAdminAnalyticsError(message){{
      ['adminPieChart','adminBarChart','adminLineChart'].forEach(id=>{{
        const canvas=document.getElementById(id);
        if(!canvas) return;
        const wrapper=canvas.parentElement;
        wrapper.innerHTML = `<div class="empty">${{message}}</div>`;
      }});
    }}

    fetch('/api/admin-analytics')
      .then(r=>r.json())
      .then(data=>{{
        // Risk Distribution Pie Chart
        new Chart(document.getElementById('adminPieChart'),{{
          type:'doughnut',
          data:{{
            labels:['High Risk','Medium Risk','Low Risk'],
            datasets:[{{
              data:[data.risk_high||0,data.risk_medium||0,data.risk_low||0],
              backgroundColor:['#ef4444','#f97316','#22c55e'],
              borderWidth:2,borderColor:'#fff'
            }}]
          }},
          options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'bottom'}}}}}}
        }});
        
        // Predictions Bar Chart
        const pred_labels=data.predictions_by_day.map(d=>d.date);
        const pred_counts=data.predictions_by_day.map(d=>d.count);
        new Chart(document.getElementById('adminBarChart'),{{
          type:'bar',
          data:{{
            labels:pred_labels,
            datasets:[{{
              label:'Predictions',
              data:pred_counts,
              backgroundColor:'#3b82f6',
              borderColor:'#1e40af',
              borderWidth:1
            }}]
          }},
          options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}}}}
        }});
        
        // Activity Line Chart
        new Chart(document.getElementById('adminLineChart'),{{
          type:'line',
          data:{{
            labels:data.users_by_day.map(d=>d.date),
            datasets:[{{
              label:'New Users',
              data:data.users_by_day.map(d=>d.count),
              borderColor:'#8b5cf6',backgroundColor:'rgba(139,92,246,.1)',
              tension:.3,fill:true,pointRadius:2
            }}]
          }},
          options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}}}}
        }});
      }})
      .catch(err=>{{
        console.error('Admin analytics failed:', err);
        showAdminAnalyticsError('Unable to load admin analytics data.');
      }});

    function editUser(id,username,name,role){{
      document.getElementById('userId').value=id;
      document.getElementById('userName').value=name;
      document.getElementById('userUsername').value=username;
      document.getElementById('userRole').value=role;
      document.getElementById('modalTitle').textContent='Edit User';
      document.getElementById('userModal').style.display='flex';
    }}

    async function saveUser(){{
      const id=document.getElementById('userId').value;
      const payload={{
        name:document.getElementById('userName').value,
        username:document.getElementById('userUsername').value,
        password:document.getElementById('userPassword').value,
        role:document.getElementById('userRole').value,
        status:document.getElementById('userStatus').value
      }};
      
      if(!payload.name||!payload.username){{
        showUserMsg('err','Name and Username are required.');return;
      }}
      
      const url=id?`/api/admin/update-user/${{id}}`:`/api/admin/add-user`;
      const res=await fetch(url,{{method:'POST',
        headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}});
      const d=await res.json();
      
      if(d.success){{
        showUserMsg('ok','User saved!');
        setTimeout(()=>location.reload(),1500);
      }}else{{
        showUserMsg('err',d.error||'Failed to save user.');
      }}
    }}

    async function deleteUser(id){{
      if(!confirm('Delete this user?'))return;
      const res=await fetch(`/api/admin/delete-user/${{id}}`,{{method:'POST'}});
      const d=await res.json();
      if(d.success){{
        showUserMsg('ok','User deleted!');
        setTimeout(()=>location.reload(),1500);
      }}else{{
        showUserMsg('err',d.error||'Failed to delete user.');
      }}
    }}

    function showUserMsg(type,text){{
      const m=document.getElementById('userMsg');
      m.className=type==='ok'?'alert alert-ok':'alert alert-err';
      m.textContent=text;
      m.style.display='block';
    }}

    function downloadReportCSV(){{
      fetch('/api/admin/report-csv').then(r=>r.blob()).then(b=>{{
        const url=URL.createObjectURL(b);
        const a=document.createElement('a');
        a.href=url;a.download='system_report.csv';
        document.body.appendChild(a);a.click();a.remove();
        URL.revokeObjectURL(url);
      }});
    }}

    function downloadUserList(){{
      fetch('/api/admin/users-csv').then(r=>r.blob()).then(b=>{{
        const url=URL.createObjectURL(b);
        const a=document.createElement('a');
        a.href=url;a.download='user_list.csv';
        document.body.appendChild(a);a.click();a.remove();
        URL.revokeObjectURL(url);
      }});
    }}

    function downloadAlertLog(){{
      fetch('/api/admin/alerts-csv').then(r=>r.blob()).then(b=>{{
        const url=URL.createObjectURL(b);
        const a=document.createElement('a');
        a.href=url;a.download='alert_log.csv';
        document.body.appendChild(a);a.click();a.remove();
        URL.revokeObjectURL(url);
      }});
    }}
    </script>
    """
    return page("Admin Dashboard", body, active="/admin")

# ══════════════════════════════════════════════════════════════════════
# SECTION 16 – ADMIN API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════
@app.route("/api/admin-analytics")
@login_required
@admin_required
def api_admin_analytics():
    """Get analytics data for admin dashboard."""
    db = get_db()
    
    # Risk distribution
    risk_data = db.execute("""
        SELECT risk_level, COUNT(*) as cnt FROM predictions GROUP BY risk_level
    """).fetchall()
    risk_dist = {d['risk_level']: d['cnt'] for d in risk_data}
    
    # Predictions by day (last 7 days)
    pred_by_day = db.execute("""
        SELECT DATE(timestamp) as date, COUNT(*) as count FROM predictions
        GROUP BY DATE(timestamp) ORDER BY date DESC LIMIT 7
    """).fetchall()
    
    # Users by day (last 7 days)
    users_by_day = db.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count FROM users
        GROUP BY DATE(created_at) ORDER BY date DESC LIMIT 7
    """).fetchall()
    
    db.close()
    
    return jsonify(dict(
        risk_high=risk_dist.get('High', 0),
        risk_medium=risk_dist.get('Medium', 0),
        risk_low=risk_dist.get('Low', 0),
        predictions_by_day=[dict(r) for r in pred_by_day][::-1],
        users_by_day=[dict(r) for r in users_by_day][::-1]
    ))

@app.route("/api/admin/add-user", methods=["POST"])
@login_required
@admin_required
def api_add_user():
    """Add a new user."""
    d = request.json
    db = get_db()
    try:
        db.execute("""INSERT INTO users (username,password,name,role,status,created_at)
            VALUES (?,?,?,?,?,?)""",
            (d['username'], d['password'], d['name'], d['role'], 
             d.get('status','Active'), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        db.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    finally:
        db.close()

@app.route("/api/admin/update-user/<int:uid>", methods=["POST"])
@login_required
@admin_required
def api_update_user(uid):
    """Update user details."""
    d = request.json
    db = get_db()
    try:
        if d.get('password'):
            db.execute("""UPDATE users SET name=?, role=?, status=?, password=? WHERE id=?""",
                (d['name'], d['role'], d.get('status','Active'), d['password'], uid))
        else:
            db.execute("""UPDATE users SET name=?, role=?, status=? WHERE id=?""",
                (d['name'], d['role'], d.get('status','Active'), uid))
        db.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    finally:
        db.close()

@app.route("/api/admin/delete-user/<int:uid>", methods=["POST"])
@login_required
@admin_required
def api_delete_user(uid):
    """Delete a user."""
    db = get_db()
    try:
        db.execute("DELETE FROM users WHERE id=?", (uid,))
        db.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    finally:
        db.close()

@app.route("/api/admin/report-csv")
@login_required
@admin_required
def api_report_csv():
    """Generate and download system report as CSV."""
    db = get_db()
    
    # Get patients data
    patients = db.execute("SELECT * FROM patients").fetchall()
    # Get predictions summary
    preds = db.execute("""
        SELECT patient_id, COUNT(*) as total, 
        SUM(CASE WHEN risk_level='High' THEN 1 ELSE 0 END) as high_risk,
        SUM(CASE WHEN fall_event=1 THEN 1 ELSE 0 END) as falls
        FROM predictions GROUP BY patient_id
    """).fetchall()
    
    db.close()
    
    # Build CSV
    csv_data = "Patient ID,Name,Age,Room Type,Assigned To,Total Predictions,High Risk,Falls\n"
    for p in patients:
        pred = next((pr for pr in preds if pr['patient_id']==p['patient_id']), None)
        csv_data += f"{p['patient_id']},{p['name']},{p['age']},{p['room_type']},{p['assigned_to']}," \
                   f"{pred['total'] if pred else 0},{pred['high_risk'] if pred else 0}," \
                   f"{pred['falls'] if pred else 0}\n"
    
    return __import__('flask').Response(csv_data, mimetype='text/csv', 
                                       headers={"Content-Disposition":"attachment;filename=system_report.csv"})

@app.route("/api/admin/users-csv")
@login_required
@admin_required
def api_users_csv():
    """Download user list as CSV."""
    db = get_db()
    users = db.execute("SELECT username,name,role,status,created_at FROM users").fetchall()
    db.close()
    
    csv_data = "Username,Name,Role,Status,Created Date\n"
    for u in users:
        csv_data += f"{u['username']},{u['name']},{u['role']},{u['status']},{u['created_at']}\n"
    
    return __import__('flask').Response(csv_data, mimetype='text/csv',
                                       headers={"Content-Disposition":"attachment;filename=user_list.csv"})

@app.route("/api/admin/alerts-csv")
@login_required
@admin_required
def api_alerts_csv():
    """Download alert log as CSV."""
    db = get_db()
    alerts = db.execute("""
        SELECT patient_id, risk_level, fall_event, fall_severity, timestamp 
        FROM predictions WHERE risk_level='High' OR fall_event=1 ORDER BY timestamp DESC
    """).fetchall()
    db.close()
    
    csv_data = "Patient ID,Risk Level,Fall Detected,Severity,Timestamp\n"
    for a in alerts:
        csv_data += f"{a['patient_id']},{a['risk_level']},{'Yes' if a['fall_event'] else 'No'},{a['fall_severity'] or 'N/A'},{a['timestamp']}\n"
    
    return __import__('flask').Response(csv_data, mimetype='text/csv',
                                      headers={"Content-Disposition":"attachment;filename=alert_log.csv"})

def cleanup_duplicate_predictions():
    """Remove duplicate demo predictions, keeping only the most recent for each patient."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Keep only the most recent prediction for each patient (based on timestamp)
    c.execute("""
        DELETE FROM predictions 
        WHERE id NOT IN (
            SELECT MAX(id) 
            FROM predictions 
            GROUP BY patient_id, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, 
                     heart_rate, room_temp, room_light, room_type, posture, time_of_day
        )
    """)
    
    deleted_count = c.rowcount
    conn.commit()
    conn.close()
    print(f"[CLEANUP] Removed {deleted_count} duplicate predictions")

# This runs init_db for gunicorn (outside __main__)
init_db()
cleanup_duplicate_predictions()

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0",
            port=int(os.environ.get("PORT", 5000)))
