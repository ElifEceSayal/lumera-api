
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from uuid import uuid4
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import os
import gdown

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1f8X-9ow2hI9sTuasydmBPqIDxyK_d_YE?usp=sharing"
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

def download_model_if_missing():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH):
        print("Model dosyaları zaten mevcut.")
        return

    print("Model Google Drive'dan indiriliyor...")
    gdown.download_folder(
        DRIVE_FOLDER_URL,
        output=MODEL_DIR,
        quiet=False,
        use_cookies=False
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config bulunamadı: {CONFIG_PATH}")

download_model_if_missing()

# ─────────────────────────────────────────────
# 0) SABITLER
# ─────────────────────────────────────────────
SAVE_DIR = Path("saved_model")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

URGENCY_SCORE  = {"KRITIK": 1.00, "YUKSEK": 0.75, "ORTA": 0.45, "DUSUK": 0.20}
SOURCE_TRUST   = {
    "CALLCENTER": 0.90, "FIELD": 0.85,
    "FORM": 0.70, "WHATSAPP": 0.50, "TWITTER": 0.40,"MOBILE_APP": 0.65,
}
LOCATION_SCORE = {
    "full": 1.00, "district_only": 0.70,
    "city_only": 0.40, "none": 0.15,
}
KRITIK_KEYWORDS = [
    "enkaz", "mahsur", "ses geliyor", "hayatta", "kan durmuyor",
    "bilinç", "nefes", "kalp", "kritik", "hayati",
    "çocuk", "bebek", "yaşlı", "hamile", "yangın", "patlama",
    "gaz", "çöktü", "yıkıldı", "altında", "kurtarma"
]
NEED_KEYWORDS = {
    "YARALI_SAGLIK": [
        "solunum cihazı", "solunum cihazi",
        "oksijen tüpü", "oksijen tupu",
        "oksijen lazım", "oksijen lazim",
        "ventilatör", "ventilator",
        "nefes alamıyor", "nefes alamiyor",
        "nefes darlığı", "nefes darligi",
        "ambulans", "yaralı", "yarali",
        "doktor", "hastane", "kanama",
        "bilinci kapalı", "bilinci kapali"
    ],
    "ILAC": [
        "ilaç", "ilac",
        "insülin", "insulin",
        "reçete", "recete",
        "serum", "antibiyotik"
    ],
    "BARINMA": [
        "çadır", "cadir",
        "barınma", "barinma",
        "kalacak yer", "evsiz", "konteyner"
    ],
    "GIDA": [
        "gıda", "gida",
        "yemek", "mama", "ekmek",
        "açız", "aciz", "erzak"
    ],
    "SU": [
        "su lazım", "su lazim",
        "içme suyu", "icme suyu",
        "susuz", "su yok"
    ],
    "ISINMA": [
        "battaniye", "ısıtıcı", "isitici",
        "üşüyoruz", "usuyoruz",
        "soğuk", "soguk",
        "ısınma", "isinma"
    ],
    "ELEKTRIK": [
        "elektrik", "şarj", "sarj",
        "jeneratör", "jenerator",
        "enerji", "karanlık", "karanlik"
    ],
    "ILETISIM": [
        "telefon", "internet",
        "hat yok", "iletişim", "iletisim",
        "şebeke", "sebeke"
    ],
    "ULASIM": [
        "yol kapalı", "yol kapali",
        "ulaşım", "ulasim",
        "köprü", "kopru",
        "trafik", "araç lazım", "arac lazim"
    ],
    "ARAMA_KURTARMA_EKIPMAN": [
        "arama kurtarma",
        "ekip lazım", "ekip lazim",
        "vinç", "vinc",
        "kepçe", "kepce",
        "kazma", "kurtarma ekibi"
    ],
    "ENKAZ": [
        "enkaz", "mahsur", "göçük", "gocuk",
        "altında kaldı", "altinda kaldi",
        "bina yıkıldı", "bina yikildi"
    ],
}
URGENCY_KEYWORDS = {
    "KRITIK": [
        "mahsur", "enkaz", "göçük", "gocuk",
        "altında kaldı", "altinda kaldi",
        "nefes alamıyor", "nefes alamiyor",
        "bilinci kapalı", "bilinci kapali",
        "kan durmuyor",
        "bebek", "çocuk", "cocuk",
        "hamile", "yaşlı", "yasli",
        "patlama", "gaz kaçağı", "gaz kacagi",
        "yangın içinde", "yangin icinde",
        "içeride kaldı", "iceride kaldi"
    ],
    "YUKSEK": [
        "yangın", "yangin",
        "alev", "duman",
        "yanıyor", "yaniyor",
        "gaz kokusu",
        "yaralı", "yarali",
        "ambulans",
        "solunum cihazı", "solunum cihazi",
        "oksijen", "ventilatör", "ventilator",
        "ilaç lazım", "ilac lazim",
        "çadır lazım", "cadir lazim",
        "su lazım", "su lazim",
        "yol kapalı", "yol kapali"
    ]
}


def rule_based_urgency(text):
    text = text.lower().strip()

    for kw in URGENCY_KEYWORDS["KRITIK"]:
        if kw in text:
            return "KRITIK"

    for kw in URGENCY_KEYWORDS["YUKSEK"]:
        if kw in text:
            return "YUKSEK"

    return None


def rule_based_need_type(text):
    text = text.lower().strip()

    for need, keywords in NEED_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return need

    return None

# Optimize edilmiş ağırlıklar
ALPHA  = 0.45
BETA   = 0.40
GAMMA  = 0.10
DELTA  = 0.05
LAMBDA = 0.05

# DBSCAN parametreleri
DBSCAN_EPS         = 0.35
DBSCAN_MIN_SAMPLES = 2
W_TEXT             = 0.55
W_LOCATION         = 0.30
W_TIME             = 0.15

# ─────────────────────────────────────────────
# 1) BERT MODEL MİMARİSİ
# ─────────────────────────────────────────────
class AfetMultiTaskModel(nn.Module):
    def __init__(self, model_name, n_need, n_urgency, dropout):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(model_name)
        hidden       = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.need_head = nn.Sequential(
            nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),   nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, n_need),
        )
        self.urgency_head = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),   nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_urgency),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden + 4, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1),   nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask,
                source_trust, info_completeness, has_hazard, actionable):
        out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls  = self.dropout(out.last_hidden_state[:, 0, :])
        need = self.need_head(cls)
        urg  = self.urgency_head(cls)
        ext  = torch.stack([source_trust, info_completeness, has_hazard, actionable], dim=1)
        risk = self.risk_head(torch.cat([cls, ext], dim=1)).squeeze(1)
        return need, urg, risk

# ─────────────────────────────────────────────
# 2) MODELLERİ YÜKLE
# ─────────────────────────────────────────────
print(f"Cihaz: {DEVICE}")
print("BERT modeli yükleniyor...")

with open(SAVE_DIR / "config.json", encoding="utf-8") as f:
    cfg = json.load(f)

# Eğitimde kullanılan gerçek sınıf sırasını config.json'dan al.
# Böylece modelin tahmin ettiği indeks doğru etikete çevrilir.
NEED_TYPES = cfg["need_classes"]
URGENCY_LEVELS = cfg["urg_classes"]

tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
bert_model = AfetMultiTaskModel(
    model_name=cfg["model_name"],
    n_need=cfg["n_need"],
    n_urgency=cfg["n_urgency"],
    dropout=cfg["dropout"],
).to(DEVICE)
bert_model.load_state_dict(torch.load(SAVE_DIR / "best_model.pt", map_location=DEVICE))
bert_model.eval()
print("BERT modeli hazır!")

print("Sentence-BERT yükleniyor...")
sbert = SentenceTransformer("sbert_model")
print("Sentence-BERT hazır!\n")

# ─────────────────────────────────────────────
# 3) VERİTABANI (in-memory)
# ─────────────────────────────────────────────
# Tüm mesajlar
MESSAGES_DB = {}   # message_id → message dict
# Tüm olaylar
EVENTS_DB   = {}   # event_id → event dict
# SBERT embedding cache
EMBEDDINGS  = {}   # message_id → embedding (numpy array)
CLIENT_MESSAGE_INDEX = {}

# ─────────────────────────────────────────────
# 4) RİSK SKORU FONKSİYONLARI
# ─────────────────────────────────────────────
def parse_time(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z", ""))
    except:
        return datetime.utcnow()

def content_score(texts):
    all_text = " ".join(t.lower() for t in texts)
    hits = sum(1 for kw in KRITIK_KEYWORDS if kw in all_text)
    return min(1.0, hits / (len(KRITIK_KEYWORDS) / 3))

def time_decay_score(timestamps):
    now    = datetime.utcnow()
    times  = [parse_time(ts) for ts in timestamps]
    latest = max(times)
    hours  = (now - latest).total_seconds() / 3600
    return math.exp(-LAMBDA * hours)

def density_score(n, max_n=20):
    return min(1.0, math.log1p(n) / math.log1p(max_n))

def compute_full_risk(messages_list):
    texts      = [m["text"] for m in messages_list]
    urgencies  = [m.get("urgency", "ORTA") for m in messages_list]
    sources    = [m.get("source", "WHATSAPP") for m in messages_list]
    timestamps = [m.get("created_at", datetime.utcnow().isoformat()) for m in messages_list]
    precisions = [m.get("location_precision", "none") for m in messages_list]

    u  = max(URGENCY_SCORE.get(urg, 0.45) for urg in urgencies)
    a  = density_score(len(messages_list))
    m  = content_score(texts)
    t  = time_decay_score(timestamps)
    l  = max(LOCATION_SCORE.get(p, 0.15) for p in precisions)
    tr = sum(SOURCE_TRUST.get(s, 0.5) for s in sources) / len(sources)

    risk = (
        0.42 * u +
        0.13 * a +
        0.25 * m +
        0.10 * t +
        0.06 * l +
        0.04 * tr
    )

    all_text = " ".join(texts).lower()

    if any(kw in all_text for kw in [
        "enkaz altında", "mahsur", "ses geliyor",
        "bina yıkıldı", "bina yikildi",
        "göçük", "gocuk", "altında kaldı", "altinda kaldi"
    ]):
        risk += 0.12

    if any(kw in all_text for kw in [
        "nefes alamıyor", "nefes alamiyor",
        "kan durmuyor", "bilinci kapalı", "bilinci kapali",
        "solunum cihazı", "solunum cihazi",
        "oksijen", "ambulans", "yaralı", "yarali"
    ]):
        risk += 0.10

    if any(kw in all_text for kw in [
        "yangın", "yangin", "patlama",
        "gaz kaçağı", "gaz kacagi",
        "duman", "alev", "yanıyor", "yaniyor"
    ]):
        risk += 0.08

    risk = min(1.0, max(0.0, risk))

    unique_src  = len(set(sources))
    n_dup       = sum(1 for msg in messages_list if msg.get("is_duplicate", False))
    dup_penalty = (n_dup / len(messages_list)) * 0.25

    confidence  = min(1.0, max(0.0,
        0.35 * tr + 0.35 * l + 0.30 * min(1.0, unique_src / 3) - dup_penalty
    ))

    priority = min(1.0, 0.50 * risk + 0.20 * confidence + 0.25 * l + 0.05 * t)

    if risk >= 0.75:
        label = "KRİTİK"
    elif risk >= 0.55:
        label = "YÜKSEK"
    elif risk >= 0.35:
        label = "ORTA"
    else:
        label = "DÜŞÜK"

    return {
        "risk_score":       round(risk, 4),
        "risk_label":       label,
        "confidence_score": round(confidence, 4),
        "priority_score":   round(priority, 4),
        "details": {
            "U_urgency":  round(u, 4),
            "A_density":  round(a, 4),
            "M_content":  round(m, 4),
            "T_time":     round(t, 4),
            "L_location": round(l, 4),
            "T_trust":    round(tr, 4),
        }
    }

# ─────────────────────────────────────────────
# 5) DBSCAN TEKİLLEŞTİRME
# ─────────────────────────────────────────────
def get_embedding(text):
    """SBERT embedding üret, cache'le."""
    key = text[:100]
    if key not in EMBEDDINGS:
        emb = sbert.encode([text], normalize_embeddings=True)[0]
        EMBEDDINGS[key] = emb
    return EMBEDDINGS[key]

def find_matching_event(new_msg):
  
    if not EVENTS_DB:
        return None

    new_emb = get_embedding(new_msg["text"])
    new_lat  = new_msg.get("lat") or 37.0
    new_lng  = new_msg.get("lng") or 37.5
    new_time = parse_time(new_msg.get("created_at", datetime.utcnow().isoformat()))

    # Zaman normalizasyon
    time_window = 24 * 3600   # 24 saat

    best_event_id = None
    best_score    = float("inf")

    for event_id, event in EVENTS_DB.items():
        if event.get("status") in ["ÇÖZÜLDÜ", "YANLIŞ_İHBAR"]:
            continue

        # Need type farklıysa atla
        if event.get("need_type") != new_msg.get("need_type"):
            continue

        # Olaydaki mesajların embedding ortalaması
        event_msgs = event.get("messages", [])
        if not event_msgs:
            continue

        event_embs = [get_embedding(m["text"]) for m in event_msgs]
        event_emb  = np.mean(event_embs, axis=0)
        event_emb  = event_emb / (np.linalg.norm(event_emb) + 1e-8)

        # Metin benzerliği (cosine distance)
        text_dist = 1 - float(np.dot(new_emb, event_emb))

        # Konum mesafesi
        event_lats = [m.get("lat") or 37.0 for m in event_msgs]
        event_lngs = [m.get("lng") or 37.5 for m in event_msgs]
        event_lat  = sum(event_lats) / len(event_lats)
        event_lng  = sum(event_lngs) / len(event_lngs)

        dlat = abs(new_lat - event_lat) / 2.0
        dlng = abs(new_lng - event_lng) / 2.0
        loc_dist = min(1.0, math.sqrt(dlat**2 + dlng**2))

        # Zaman mesafesi
        event_times = [parse_time(m.get("created_at", "")) for m in event_msgs]
        latest_event_time = max(event_times)
        time_diff = abs((new_time - latest_event_time).total_seconds())
        time_dist = min(1.0, time_diff / time_window)

        # Birleşik mesafe
        combined = W_TEXT * text_dist + W_LOCATION * loc_dist + W_TIME * time_dist

        if combined < best_score:
            best_score    = combined
            best_event_id = event_id

    # Eşik: DBSCAN_EPS altındaysa aynı olay
    if best_score < DBSCAN_EPS:
        return best_event_id
    return None

def update_event(event_id, new_msg, pred):
    """Mevcut olaya yeni mesaj ekle, skorları güncelle."""
    event = EVENTS_DB[event_id]
    event["messages"].append(new_msg)
    event["message_count"] = len(event["messages"])
    event["updated_at"]    = datetime.utcnow().isoformat() + "Z"

    # Risk skorunu güncelle
    risk_data = compute_full_risk(event["messages"])
    event["risk_score"]       = risk_data["risk_score"]
    event["risk_label"]       = risk_data["risk_label"]
    event["confidence_score"] = risk_data["confidence_score"]
    event["priority_score"]   = risk_data["priority_score"]

    # En yüksek urgency'i koru
    urg_order = ["KRITIK", "YUKSEK", "ORTA", "DUSUK"]
    current_idx = urg_order.index(event.get("urgency", "ORTA"))
    new_idx     = urg_order.index(pred["urgency"]) if pred["urgency"] in urg_order else 2
    if new_idx < current_idx:
        event["urgency"] = pred["urgency"]

    return event

def create_new_event(msg, pred, location):
    """Yeni olay oluştur."""
    event_id  = str(uuid4())
    risk_data = compute_full_risk([msg])

    # Karar notu
    reasons = []
    if pred["need_confidence"] >= 0.85:
        reasons.append(f"{pred['need_type']} ihtiyacı tespit edildi")
    if pred["urgency"] in ["KRITIK", "YUKSEK"]:
        reasons.append(f"Aciliyet: {pred['urgency']}")
    if risk_data["details"]["M_content"] > 0.3:
        reasons.append("Kritik kelimeler var")
    if location.get("precision") == "none":
        reasons.append("Konum eksik")

    event = {
        "event_id":         event_id,
        "need_type":        pred["need_type"],
        "urgency":          pred["urgency"],
        "risk_score":       risk_data["risk_score"],
        "risk_label":       risk_data["risk_label"],
        "confidence_score": risk_data["confidence_score"],
        "priority_score":   risk_data["priority_score"],
        "message_count":    1,
        "messages":         [msg],
        "location":         location,
        "decision_note":    " | ".join(reasons) if reasons else "Standart önceliklendirme",
        "status":           "YENI",
        "authority_status": "YENI",
        "authority_decision": None,
        "authority_note": "",
        "assigned_team": "",
        "decision_by": "",
        "decision_at": None,
        "history": [
    {
        "time": datetime.utcnow().isoformat() + "Z",
        "action": "AI_EVENT_CREATED",
        "note": "Olay yapay zekâ tarafından oluşturuldu.",
        "ai_need_type": pred["need_type"],
        "ai_urgency": pred["urgency"],
        "ai_risk_score": risk_data["risk_score"],
    }
],
        "created_at":       datetime.utcnow().isoformat() + "Z",
        "updated_at":       datetime.utcnow().isoformat() + "Z",
    }
    EVENTS_DB[event_id] = event
    return event

# ─────────────────────────────────────────────
# 6) BERT TAHMİN FONKSİYONU
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_message(text, source_trust=0.5, info_completeness=0.5,
                    has_hazard=0.0, actionable=1.0):
    enc  = tokenizer(text, max_length=128, padding="max_length",
                     truncation=True, return_tensors="pt")
    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    def t(v):
        return torch.tensor([v], dtype=torch.float).to(DEVICE)

    need_logits, urg_logits, risk = bert_model(
        ids, mask, t(source_trust), t(info_completeness), t(has_hazard), t(actionable)
    )

    need_probs = torch.softmax(need_logits, dim=1)[0].cpu().tolist()
    urg_probs  = torch.softmax(urg_logits,  dim=1)[0].cpu().tolist()
    need_idx   = int(torch.argmax(need_logits))
    urg_idx    = int(torch.argmax(urg_logits))

    return {
        "need_type":       NEED_TYPES[need_idx],
        "need_confidence": round(need_probs[need_idx], 4),
        "urgency":         URGENCY_LEVELS[urg_idx],
        "urg_confidence":  round(urg_probs[urg_idx], 4),
        "need_probs":      {c: round(p, 4) for c, p in zip(NEED_TYPES, need_probs)},
    }

# ─────────────────────────────────────────────
# 7) FASTAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title="Lumera — Afet Karar Destek Sistemi",
    description="TEKNOFEST 2025",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    client_message_id:  Optional[str] = None
    text:               str
    source:             str            = "MOBILE_APP"
    location_precision: str            = "none"
    lat:                Optional[float] = None
    lng:                Optional[float] = None
    city:               str            = ""
    district:           str            = ""
    neighborhood:       str            = ""
    address:            str            = ""
    created_at_client:  Optional[str] = None

class AuthorityDecisionRequest(BaseModel):
    status: str
    decision: Optional[str] = None
    note: Optional[str] = ""
    assigned_team: Optional[str] = ""
    decision_by: Optional[str] = "Yetkili Kullanıcı"
    corrected_need_type: Optional[str] = None
    corrected_urgency: Optional[str] = None
    corrected_priority_score: Optional[float] = None

class MobileSyncRequest(BaseModel):
    reports: list[PredictRequest]

# ─────────────────────────────────────────────
# 8) ENDPOINT'LER
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "sistem":    "Lumera Afet Karar Destek Sistemi",
        "versiyon":  "2.0.0",
        "durum":     "aktif",
        "endpoints": ["/predict", "/events", "/events/{id}", "/stats"],
    }

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Ana endpoint.
    1. BERT ile mesajı analiz et
    2. DBSCAN ile mevcut olaya benziyor mu kontrol et
    3. Benziyorsa o olaya ekle, yoksa yeni olay oluştur
    4. Güncel risk/güven/öncelik skorlarını döndür
    """
        # Mobil uygulama offline-first çalıştığı için aynı bildirim tekrar gönderilebilir.
    # client_message_id daha önce işlendi ise yeniden olay oluşturma.
    if req.client_message_id and req.client_message_id in CLIENT_MESSAGE_INDEX:
        old_message_id = CLIENT_MESSAGE_INDEX[req.client_message_id]
        old_msg = MESSAGES_DB.get(old_message_id)

        if old_msg:
            old_event_id = old_msg.get("event_id")
            old_event = EVENTS_DB.get(old_event_id)

            return {
                "duplicate_client_message": True,
                "message": "Bu mobil bildirim daha önce işlenmiş.",
                "client_message_id": req.client_message_id,
                "message_id": old_message_id,
                "event_id": old_event_id,
                "need_type": old_msg.get("need_type"),
                "urgency": old_msg.get("urgency"),
                "risk_score": old_event.get("risk_score") if old_event else None,
                "risk_label": old_event.get("risk_label") if old_event else None,
                "authority_status": old_event.get("authority_status") if old_event else None,
            }
        
    src_trust  = SOURCE_TRUST.get(req.source, 0.5)
    loc_score  = LOCATION_SCORE.get(req.location_precision, 0.15)
    actionable = 1.0 if req.location_precision in ["full", "district_only"] else 0.0

    # BERT tahmini
    pred = predict_message(
    text=req.text,
    source_trust=src_trust,
    info_completeness=loc_score,
    actionable=actionable,
)

# Kural tabanlı need_type düzeltmesi
    rule_need = None

    if "rule_based_need_type" in globals():
     rule_need = rule_based_need_type(req.text)

    if rule_need:
        pred["need_type"] = rule_need
        pred["need_confidence"] = max(pred.get("need_confidence", 0), 0.95)
        pred["need_probs"] = {c: 0.0 for c in NEED_TYPES}
        pred["need_probs"][rule_need] = pred["need_confidence"]


# Kural tabanlı urgency düzeltmesi
    rule_urgency = None

    if "rule_based_urgency" in globals():
        rule_urgency = rule_based_urgency(req.text)

    if rule_urgency:
        pred["urgency"] = rule_urgency
        pred["urg_confidence"] = max(pred.get("urg_confidence", 0), 0.95)

    # Konum
    location = {
        "city":         req.city,
        "district":     req.district,
        "neighborhood": req.neighborhood,
        "address":      req.address,
        "precision":    req.location_precision,
        "lat":          req.lat,
        "lng":          req.lng,
    }

    # Mesaj kaydı
    msg_id = str(uuid4())
    created_at_value = req.created_at_client or datetime.utcnow().isoformat() + "Z"

    msg = {
        "message_id":         msg_id,
        "client_message_id":  req.client_message_id,
        "text":               req.text,
        "source":             req.source,
        "need_type":          pred["need_type"],
        "urgency":            pred["urgency"],
        "location_precision": req.location_precision,
        "lat":                req.lat or 37.0,
        "lng":                req.lng or 37.5,
        "created_at":         created_at_value,
        "received_at":        datetime.utcnow().isoformat() + "Z",
        "is_duplicate":       False,
        "event_id":           None,
    }
    MESSAGES_DB[msg_id] = msg

    # DBSCAN: mevcut olaya ekle mi, yeni olay oluştur mu?
    matching_event_id = find_matching_event(msg)

    if matching_event_id:
        event         = update_event(matching_event_id, msg, pred)
        is_duplicate  = True
        action        = "MEVCUT_OLAYA_EKLENDI"
    else:
        event         = create_new_event(msg, pred, location)
        is_duplicate  = False
        action        = "YENI_OLAY_OLUSTURULDU"
        msg["is_duplicate"] = False
    msg["event_id"] = event["event_id"]

    if req.client_message_id:
        CLIENT_MESSAGE_INDEX[req.client_message_id] = msg_id

    return {
        # Mesaj analizi
        "message_id":       msg_id,
        "need_type":        pred["need_type"],
        "need_confidence":  pred["need_confidence"],
        "urgency":          pred["urgency"],
        "urg_confidence":   pred["urg_confidence"],
        "top3_need":        dict(sorted(pred["need_probs"].items(),
                                        key=lambda x: -x[1])[:3]),
        # Olay bilgisi
        "event_id":         event["event_id"],
        "action":           action,
        "is_duplicate":     is_duplicate,
        "event_message_count": event["message_count"],
        # Skorlar
        "risk_score":       event["risk_score"],
        "risk_label":       event["risk_label"],
        "confidence_score": event["confidence_score"],
        "priority_score":   event["priority_score"],
        "decision_note":    event["decision_note"],
        # Konum
        "location":         location,
        "rule_based_need_type": rule_need,
        "rule_based_urgency": rule_urgency,
        "client_message_id": req.client_message_id,
        "sync_status": "SYNCED",
        "authority_status": event.get("authority_status", event.get("status", "YENI")),
    }

@app.get("/events")
def get_events(
    limit:      int            = 50,
    risk_label: Optional[str]  = None,
    need_type:  Optional[str]  = None,
    status:     Optional[str]  = None,
):
    """Tüm olayları öncelik sırasıyla getir."""
    events = list(EVENTS_DB.values())

    if risk_label:
        events = [e for e in events if e.get("risk_label") == risk_label]
    if need_type:
        events = [e for e in events if e.get("need_type") == need_type]
    if status:
        events = [e for e in events if e.get("status") == status]

    events.sort(key=lambda e: e.get("priority_score", 0), reverse=True)

    # Harita için sadece koordinat içerenleri döndür
    map_pins = []
    for e in events:
        msgs_with_loc = [m for m in e.get("messages", [])
                         if m.get("lat") and m.get("lng")]
        if msgs_with_loc:
            latest = msgs_with_loc[-1]
            map_pins.append({
                "event_id":     e["event_id"],
                "need_type":    e["need_type"],
                "risk_label":   e["risk_label"],
                "priority":     e["priority_score"],
                "lat":          latest["lat"],
                "lng":          latest["lng"],
                "message_count": e["message_count"],
            })

    return {
        "total":    len(events),
        "events":   [{ k: v for k, v in e.items() if k != "messages" }
                     for e in events[:limit]],
        "map_pins": map_pins,
    }

@app.get("/events/{event_id}")
def get_event(event_id: str):
    """Tek olayın detayını getir."""
    if event_id not in EVENTS_DB:
        raise HTTPException(status_code=404, detail="Olay bulunamadı")
    return EVENTS_DB[event_id]

@app.patch("/events/{event_id}/status")
def update_status(event_id: str, body: dict):
    """Olay durumunu güncelle."""
    if event_id not in EVENTS_DB:
        raise HTTPException(status_code=404, detail="Olay bulunamadı")

    valid  = ["YENİ", "DOĞRULANDI", "EKİP_YÖNLENDİRİLDİ",
              "MÜDAHALE_SÜRÜYOR", "ÇÖZÜLDÜ", "YANLIŞ_İHBAR"]
    status = body.get("status", "")
    if status not in valid:
        raise HTTPException(status_code=400, detail=f"Geçersiz durum: {valid}")

    EVENTS_DB[event_id]["status"]     = status
    EVENTS_DB[event_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    return {"event_id": event_id, "status": status}

@app.patch("/events/{event_id}/authority-decision")
def authority_decision(event_id: str, body: AuthorityDecisionRequest):
    if event_id not in EVENTS_DB:
        raise HTTPException(status_code=404, detail="Olay bulunamadı")

    event = EVENTS_DB[event_id]

    valid_status = [
        "YENI",
        "INCELENIYOR",
        "DOGRULANDI",
        "EKIP_YONLENDIRILDI",
        "MUDAHALE_SURUYOR",
        "COZULDU",
        "YANLIS_IHBAR"
    ]

    valid_decisions = [
        "SAHA_EKIBI_GONDER",
        "SAGLIK_EKIBI_GONDER",
        "ARAMA_KURTARMA_GONDER",
        "LOJISTIK_DESTEK_GONDER",
        "EK_BILGI_ISTE",
        "ONCELIGI_ARTIR",
        "ONCELIGI_DUSUR",
        "YANLIS_IHBAR",
        None
    ]

    if body.status not in valid_status:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz status. Geçerli değerler: {valid_status}"
        )

    if body.decision not in valid_decisions:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz karar. Geçerli değerler: {valid_decisions}"
        )

    old_status = event.get("authority_status", event.get("status", "YENI"))

    event["authority_status"] = body.status
    event["status"] = body.status
    event["authority_decision"] = body.decision
    event["authority_note"] = body.note or ""
    event["assigned_team"] = body.assigned_team or ""
    event["decision_by"] = body.decision_by or "Yetkili Kullanıcı"
    event["decision_at"] = datetime.utcnow().isoformat() + "Z"
    event["updated_at"] = datetime.utcnow().isoformat() + "Z"

    if body.corrected_need_type:
        event["need_type"] = body.corrected_need_type

    if body.corrected_urgency:
        event["urgency"] = body.corrected_urgency

    if body.corrected_priority_score is not None:
        event["priority_score"] = max(0.0, min(1.0, body.corrected_priority_score))

    if "history" not in event:
        event["history"] = []

    event["history"].append({
        "time": datetime.utcnow().isoformat() + "Z",
        "action": "AUTHORITY_DECISION",
        "old_status": old_status,
        "new_status": body.status,
        "decision": body.decision,
        "note": body.note,
        "assigned_team": body.assigned_team,
        "decision_by": body.decision_by,
        "corrected_need_type": body.corrected_need_type,
        "corrected_urgency": body.corrected_urgency,
        "corrected_priority_score": body.corrected_priority_score,
    })

    return {
        "event_id": event_id,
        "status": event["authority_status"],
        "decision": event["authority_decision"],
        "note": event["authority_note"],
        "assigned_team": event["assigned_team"],
        "decision_by": event["decision_by"],
        "decision_at": event["decision_at"],
        "updated_event": {
            "need_type": event["need_type"],
            "urgency": event["urgency"],
            "risk_score": event["risk_score"],
            "priority_score": event["priority_score"],
            "message_count": event["message_count"],
        }
    }

@app.get("/stats")
def get_stats():
    """Dashboard istatistikleri."""
    events = list(EVENTS_DB.values())
    if not events:
        return {"total_events": 0, "total_messages": 0}

    risk_dist   = Counter(e.get("risk_label", "DÜŞÜK") for e in events)
    need_dist   = Counter(e.get("need_type",  "DIGER")  for e in events)
    status_dist = Counter(e.get("status",     "YENİ")   for e in events)

    total_msgs  = sum(e.get("message_count", 1) for e in events)
    dedup_count = total_msgs - len(events)

    return {
        "total_events":        len(events),
        "total_messages":      total_msgs,
        "deduplicated":        dedup_count,
        "dedup_rate":          round(dedup_count / total_msgs * 100, 1) if total_msgs > 0 else 0,
        "critical_count":      risk_dist.get("KRİTİK", 0),
        "risk_distribution":   dict(risk_dist),
        "need_distribution":   dict(need_dist),
        "status_distribution": dict(status_dist),
        "avg_priority":        round(sum(e.get("priority_score", 0) for e in events) / len(events), 4),
    }
@app.get("/mobile/messages/{client_message_id}")
def get_mobile_message_status(client_message_id: str):
    if client_message_id not in CLIENT_MESSAGE_INDEX:
        raise HTTPException(status_code=404, detail="Mobil bildirim bulunamadı")

    message_id = CLIENT_MESSAGE_INDEX[client_message_id]
    msg = MESSAGES_DB.get(message_id)

    if not msg:
        raise HTTPException(status_code=404, detail="Mesaj bulunamadı")

    event_id = msg.get("event_id")
    event = EVENTS_DB.get(event_id)

    return {
        "client_message_id": client_message_id,
        "message_id": message_id,
        "event_id": event_id,
        "text": msg.get("text"),
        "need_type": msg.get("need_type"),
        "urgency": msg.get("urgency"),
        "created_at": msg.get("created_at"),
        "received_at": msg.get("received_at"),
        "authority_status": event.get("authority_status") if event else None,
        "authority_decision": event.get("authority_decision") if event else None,
        "authority_note": event.get("authority_note") if event else None,
        "assigned_team": event.get("assigned_team") if event else None,
        "decision_at": event.get("decision_at") if event else None,
        "risk_score": event.get("risk_score") if event else None,
        "risk_label": event.get("risk_label") if event else None,
    }

@app.delete("/events/reset")
def reset():
    """Tüm veriyi sıfırla (test için)."""
    EVENTS_DB.clear()
    MESSAGES_DB.clear()
    EMBEDDINGS.clear()
    CLIENT_MESSAGE_INDEX.clear()
    return {"message": "Sıfırlandı"}



@app.post("/mobile/sync")
def mobile_sync(body: MobileSyncRequest):
    results = []

    for report in body.reports:
        try:
            result = predict(report)
            results.append({
                "client_message_id": report.client_message_id,
                "ok": True,
                "result": result,
            })
        except Exception as e:
            results.append({
                "client_message_id": report.client_message_id,
                "ok": False,
                "error": str(e),
            })

    return {
        "total": len(body.reports),
        "success": sum(1 for r in results if r["ok"]),
        "failed": sum(1 for r in results if not r["ok"]),
        "results": results,
    }