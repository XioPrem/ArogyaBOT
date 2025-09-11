# app.py (fixed & full)
import json
import io
import time
import re
import os
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
from cachetools import TTLCache
from PyPDF2 import PdfReader
from dateutil import parser as dateparser
from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")

# Twilio
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client

# ---- Create Flask app early so @app.route decorators work ----
app = Flask(__name__, static_folder="static", template_folder="templates")

# Optional OpenAI — use only if OPENAI_API_KEY is set as env var
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # optional

# ---- Load config (robust) ----
CFG = {}
try:
    with open("config.json", "r", encoding="utf-8") as f:
        CFG = json.load(f)
except FileNotFoundError:
    print("config.json not found — using defaults.")
except Exception as e:
    print("Error loading config.json:", e)

CACHE_TTL = CFG.get("default_ttl_seconds", 3600)
cache = TTLCache(maxsize=500, ttl=CACHE_TTL)

# Simple "seed" QA DB derived from config (optional seed_qa.json)
SEED_QA = {}
seed_path = Path(__file__).parent / "seed_qa.json"
if seed_path.exists():
    try:
        with open(seed_path, "r", encoding="utf-8") as f:
            SEED_QA = json.load(f)
    except Exception as e:
        print("Failed to load seed_qa.json:", e)
        SEED_QA = {}

def simple_match_seed_answer(question):
    """
    Try exact or fuzzy match in SEED_QA (keys are lowercase)
    """
    q = (question or "").strip().lower()
    # exact
    if q in SEED_QA:
        return SEED_QA[q]
    # partial match: check keys that are substrings
    for k, v in SEED_QA.items():
        if k in q or q in k:
            return v
    return None

def craft_rule_based_reply(user_question, disease, best_hit):
    """
    Create a short conversational reply from the structured data we have.
    """
    parts = []
    dos = best_hit.get("dos", []) if best_hit else []
    donts = best_hit.get("donts", []) if best_hit else []
    summary = (best_hit.get("summary") or "")[:1500] if best_hit else ""
    qlow = (user_question or "").lower()
    if any(w in qlow for w in ["what should i do", "what to do", "how to treat", "should i", "what to do if", "if i have"]):
        if dos:
            parts.append("Here are recommended actions:")
            for d in dos[:6]:
                parts.append(f"• {d}")
        if donts:
            parts.append("\nThings to avoid:")
            for d in donts[:6]:
                parts.append(f"• {d}")
        if not dos and summary:
            parts.append("Brief guidance: " + (summary.split("\n")[0] if summary else "Refer to official guidance."))
        parts.append("\n(For more detail I can show the PDF summary or give a longer explanation.)")
        return "\n".join(parts)
    # Generic fallback summary reply
    if summary:
        para = summary.split("\n\n")
        short = para[0] if para else summary[:300]
        return f"Summary: {short}\n\n(You can ask 'what should I do' to get Do's & Don'ts.)"
    return "Sorry — I don't have a direct answer in the seed data. Try asking 'what should I do' or enable the LLM mode (set OPENAI_API_KEY)."

def call_openai_chat(prompt, max_tokens=250):
    """
    Minimal OpenAI API call using the REST endpoint. Returns text or None.
    (This is optional. Only used if OPENAI_API_KEY env var is set.)
    """
    if not OPENAI_API_KEY:
        return None
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful medical-information assistant. Use only the provided sources. If unsure, advise to consult a healthcare professional."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2
        }
        r = requests.post(url, headers=headers, json=body, timeout=15)
        r.raise_for_status()
        j = r.json()
        if "choices" in j and len(j["choices"]) > 0:
            txt = j["choices"][0]["message"]["content"]
            return txt
        return None
    except Exception as e:
        print("OpenAI call failed:", e)
        return None

# ---- Helpers ----
def normalize(s):
    return (s or "").strip().lower()

def cached_get(key, fn):
    if key in cache:
        return cache[key]
    val = fn()
    cache[key] = val
    return val

# ---- Utility: extract bullets and lists heuristics from text ----
def extract_dos_donts_from_text(text, max_items=6):
    """
    Heuristic extraction of Do's and Don'ts from text.
    """
    if not text:
        return [], []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    dos, donts = [], []
    for i, line in enumerate(lines):
        low = line.lower()
        if re.search(r"\b(do|dos|do's|recommendations|precautions|what to do)\b", low):
            for j in range(i + 1, min(i + 12, len(lines))):
                l = lines[j]
                if re.match(r"^(\-|\u2022|\*|\d+\.)\s*", l):
                    dos.append(re.sub(r"^(\-|\u2022|\*|\d+\.)\s*", "", l))
                else:
                    if re.search(r"\b(don't|dont|avoid|warnings|what not to do)\b", l.lower()):
                        break
                    if len(l.split()) <= 20:
                        dos.append(l)
                if len(dos) >= max_items:
                    break
        if re.search(r"\b(don't|dont|avoid|warnings|what not to do)\b", low):
            for j in range(i + 1, min(i + 12, len(lines))):
                l = lines[j]
                if re.match(r"^(\-|\u2022|\*|\d+\.)\s*", l):
                    donts.append(re.sub(r"^(\-|\u2022|\*|\d+\.)\s*", "", l))
                else:
                    if len(l.split()) <= 20:
                        donts.append(l)
                if len(donts) >= max_items:
                    break
    # fallback bullets classification
    if not dos or not donts:
        bullets = [re.sub(r"^(\-|\u2022|\*|\d+\.)\s*", "", l) for l in lines if re.match(r"^(\-|\u2022|\*|\d+\.)\s*", l)]
        for b in bullets:
            low = b.lower()
            if any(k in low for k in ["avoid", "do not", "don't", "without", "not use", "not to"]):
                if len(donts) < max_items:
                    donts.append(b)
            elif any(k in low for k in ["use", "seek", "visit", "isolate", "wash", "wear", "get", "contact"]):
                if len(dos) < max_items:
                    dos.append(b)
    # last-resort keywords
    if not dos:
        for l in lines:
            low = l.lower()
            if any(k in low for k in ["seek medical", "use paracetamol", "isolate", "wear mask", "hydrate", "wash hands"]):
                dos.append(l)
            if len(dos) >= max_items:
                break
    if not donts:
        for l in lines:
            low = l.lower()
            if any(k in low for k in ["avoid", "do not", "don't", "not use", "no aspirin", "no ibuprofen", "self medicate"]):
                donts.append(l)
            if len(donts) >= max_items:
                break
    return dos[:max_items], donts[:max_items]

# ---- PDF fetcher with Dos/Don't extraction ----
def fetch_pdf_text_and_extract(url, max_chars=6000, timeout=12):
    """Download a PDF, extract first pages text and try to parse dos/donts heuristically."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        reader = PdfReader(io.BytesIO(r.content))
        meta = reader.metadata or {}
        last_mod = None
        for k in ["/ModDate", "/CreationDate", "ModDate", "CreationDate"]:
            if k in meta and meta[k]:
                last_mod = meta[k]
                break
        if last_mod:
            try:
                last_mod = dateparser.parse(str(last_mod)).isoformat()
            except Exception:
                pass
        pages_text = []
        pages_to_read = min(6, len(reader.pages))
        for i in range(pages_to_read):
            try:
                txt = reader.pages[i].extract_text() or ""
                pages_text.append(txt)
            except Exception:
                continue
        full_text = "\n".join(pages_text).strip()
        snippet = full_text[:max_chars]
        dos, donts = extract_dos_donts_from_text(full_text)
        return {"url": url, "text_snippet": snippet, "last_updated": last_mod, "raw_meta": {k: v for k, v in meta.items()}, "dos": dos, "donts": donts}
    except Exception as e:
        return {"error": str(e), "url": url}

# ---- NCDC notifications scraper ----
NCDC_NOTIF_URL = "https://ncdc.mohfw.gov.in/notifications/"

def fetch_ncdc_notifications_for(disease, max_items=5, timeout=10):
    try:
        r = requests.get(NCDC_NOTIF_URL, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        found = []
        candidates = soup.select("div.post, article, div.card, li, div.news-content") or soup.find_all(["article", "li", "div"])
        for c in candidates:
            title_tag = c.find(["h2", "h3", "h4", "a"])
            title = title_tag.get_text().strip() if title_tag else ""
            summary = ""
            p = c.find("p")
            if p:
                summary = p.get_text().strip()
            link_tag = c.find("a", href=True)
            link = link_tag['href'] if link_tag else NCDC_NOTIF_URL
            combined = (title + " " + summary).lower()
            if disease.lower() in combined:
                pubDate = None
                time_tag = c.find("time")
                if time_tag and time_tag.get("datetime"):
                    pubDate = time_tag.get("datetime")
                else:
                    strings = list(c.stripped_strings)
                    for s in strings:
                        try:
                            dt = dateparser.parse(s, fuzzy=True)
                            pubDate = dt.isoformat()
                            break
                        except Exception:
                            continue
                found.append({"title": title or disease.title(), "summary": summary, "link": link, "pubDate": pubDate})
            if len(found) >= max_items:
                break
        return found
    except Exception as e:
        return {"error": str(e)}

# ---- IDSP quick-check ----
IDSP_BASE = "https://idsp.mohfw.gov.in/"

def fetch_idsp_recent_links(timeout=10):
    try:
        r = requests.get(IDSP_BASE, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a['href']
            text = a.get_text().strip().lower()
            if any(k in text for k in ["week", "weekly", "report", "outbreak", ".pdf"]):
                url = href if href.startswith("http") else IDSP_BASE.rstrip("/") + "/" + href.lstrip("/")
                links.append({"text": a.get_text().strip(), "url": url})
        seen = set()
        out = []
        for l in links:
            if l["url"] not in seen:
                seen.add(l["url"])
                out.append(l)
            if len(out) >= 8:
                break
        return out
    except Exception as e:
        return {"error": str(e)}

# ---- Core source query logic (uses PDF extractor) ----
def find_sources_for_disease(disease):
    nd = normalize(disease)
    matches = []
    for s in CFG.get("sources", []):
        ds = [normalize(d) for d in s.get("diseases", [])]
        if nd in ds or any(nd in d for d in ds):
            matches.append(s)
    if not matches:
        matches = CFG.get("sources", [])
    return matches

def query_sources(disease, region=None):
    key = f"disease:{disease}|region:{region}"
    def _work():
        sources = find_sources_for_disease(disease)
        aggregated = {"hits": [], "queried": [], "disease": disease, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        for s in sources:
            stype = s.get("type")
            aggregated["queried"].append({"id": s.get("id"), "type": stype})
            if stype == "pdf":
                url = s.get("pdf_url")
                if url:
                    res = fetch_pdf_text_and_extract(url)
                    if res and "error" not in res:
                        res_out = {
                            "title": s.get("name"),
                            "summary": res.get("text_snippet", ""),
                            "dos": res.get("dos", []),
                            "donts": res.get("donts", []),
                            "source_url": url,
                            "last_updated": res.get("last_updated"),
                            "confidence": 0.95,
                            "source_id": s.get("id"),
                            "source_name": s.get("name")
                        }
                        aggregated["hits"].append(res_out)
            elif stype == "page":
                page_url = s.get("page_url")
                try:
                    r = requests.get(page_url, timeout=8)
                    if r.status_code == 200:
                        soup = BeautifulSoup(r.text, "html.parser")
                        text = soup.get_text(separator="\n").strip()
                        if disease.lower() in text.lower():
                            dos, donts = extract_dos_donts_from_text(text)
                            res_out = {
                                "title": s.get("name"),
                                "summary": text[:2000],
                                "dos": dos,
                                "donts": donts,
                                "source_url": page_url,
                                "last_updated": None,
                                "confidence": 0.8,
                                "source_id": s.get("id"),
                                "source_name": s.get("name")
                            }
                            aggregated["hits"].append(res_out)
                except Exception:
                    pass
            elif stype == "ncdc":
                ncdc_hits = fetch_ncdc_notifications_for(disease)
                if isinstance(ncdc_hits, dict) and ncdc_hits.get("error"):
                    pass
                else:
                    for item in ncdc_hits:
                        res_out = {
                            "title": item.get("title"),
                            "summary": item.get("summary"),
                            "dos": [],
                            "donts": [],
                            "source_url": item.get("link"),
                            "last_updated": item.get("pubDate"),
                            "confidence": 0.78,
                            "source_id": s.get("id"),
                            "source_name": s.get("name")
                        }
                        aggregated["hits"].append(res_out)
            elif stype == "idsp":
                links = fetch_idsp_recent_links()
                if isinstance(links, dict) and links.get("error"):
                    pass
                else:
                    aggregated["hits"].append({
                        "title": s.get("name"),
                        "summary": "Recent surveillance/report links (IDSP).",
                        "links": links,
                        "source_url": s.get("page_url"),
                        "last_updated": None,
                        "confidence": 0.5,
                        "source_id": s.get("id"),
                        "source_name": s.get("name")
                    })
            # break early if high-confidence hit found
            if aggregated["hits"] and any(h.get("confidence", 0) >= 0.9 for h in aggregated["hits"]):
                break
        return aggregated
    return cached_get(key, _work)

# ---- Flask endpoints ----
@app.route("/chat", methods=["POST"])
def chat():
    """
    Conversational endpoint. POST JSON:
    { "message": "...", "use_llm": true|false, "region": "Kolkata" (optional) }
    """
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    use_llm = data.get("use_llm", False)
    region = data.get("region")
    if not message:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    # 1) Try seed QA (if user has seed_qa.json)
    seed_ans = simple_match_seed_answer(message)
    if seed_ans:
        return jsonify({"ok": True, "answer": seed_ans, "source": {"type": "seed"}, "llm_used": False})

    # 2) Try to detect a disease keyword
    disease = parse_disease_from_text(message)
    query_term = disease or (message if len(message.split()) <= 4 else "")
    if not query_term:
        query_term = message.split()[-1].lower()

    # 3) Query configured sources
    try:
        results = query_sources(query_term, region)
    except Exception as e:
        print("query_sources error:", e)
        results = {"hits": []}

    best_hit = None
    if results.get("hits"):
        best_hit = sorted(results["hits"], key=lambda x: x.get("confidence", 0), reverse=True)[0]

    # 4) Craft rule-based reply
    rule_reply = craft_rule_based_reply(message, query_term, best_hit)

    # 5) Optionally call LLM
    if use_llm and OPENAI_API_KEY:
        context_parts = []
        if best_hit:
            context_parts.append(f"Source title: {best_hit.get('title')}\n")
            if best_hit.get("summary"):
                context_parts.append("Summary snippet:\n" + (best_hit.get("summary")[:2000]) + "\n")
            if best_hit.get("dos"):
                context_parts.append("Do's:\n" + "\n".join(f"- {d}" for d in best_hit.get("dos")[:10]) + "\n")
            if best_hit.get("donts"):
                context_parts.append("Don'ts:\n" + "\n".join(f"- {d}" for d in best_hit.get("donts")[:10]) + "\n")
        prompt = (
            "Use the following source material to answer the user's question as a short, clear, actionable reply. "
            "If the sources do not contain an answer, say you don't know and suggest contacting a healthcare professional.\n\n"
            f"SOURCE MATERIAL:\n{''.join(context_parts)}\n\nUSER QUESTION:\n{message}\n\nANSWER:"
        )
        llm_answer = call_openai_chat(prompt, max_tokens=300)
        if llm_answer:
            return jsonify({"ok": True, "answer": llm_answer.strip(), "source": best_hit or {}, "llm_used": True})

    # 6) Return rule-based reply
    return jsonify({"ok": True, "answer": rule_reply, "source": best_hit or {}, "llm_used": False})

@app.route("/disease_info", methods=["POST"])
def disease_info():
    data = request.get_json() or {}
    disease = data.get("disease")
    if not disease:
        return jsonify({"error": "Please provide 'disease' in request body."}), 400
    region = data.get("region")
    result = query_sources(disease, region)
    if not result["hits"]:
        return jsonify({
            "disease": disease,
            "found": False,
            "message": f"No content found for '{disease}' in configured sources.",
            "queried_sources": result["queried"],
            "fetched_at": result["fetched_at"]
        }), 200
    best = sorted(result["hits"], key=lambda x: x.get("confidence", 0), reverse=True)[0]
    resp = {
        "disease": disease,
        "found": True,
        "result": best,
        "queried_sources": result["queried"],
        "fetched_at": result["fetched_at"]
    }
    return jsonify(resp), 200

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/static/<path:p>")
def static_files(p):
    static_dir = Path(__file__).parent / "static"
    return send_from_directory(static_dir, p)

# ---- Twilio / WhatsApp integration ----
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

# Initialize Twilio client only if credentials are present (client accepts empty strings but calls will fail)
try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
except Exception:
    twilio_client = None

def parse_disease_from_text(text):
    t = (text or "").strip().lower()
    DISEASE_KEYWORDS = ["dengue", "malaria", "covid", "covid-19", "influenza", "typhoid", "chikungunya", "measles", "hepatitis", "tuberculosis"]
    for d in DISEASE_KEYWORDS:
        if re.search(rf"\b{re.escape(d)}\b", t):
            return d
    if len(t.split()) <= 3:
        return t
    m = re.search(r"about ([a-zA-Z0-9\-\s]+)", t)
    if m:
        return m.group(1).strip().split()[0]
    return None

@app.route("/whatsapp_webhook", methods=["POST"])
def whatsapp_webhook():
    from_number = request.form.get("From")
    body = request.form.get("Body", "").strip()
    resp = MessagingResponse()

    if not body:
        resp.message("Hi! Send a disease name (e.g., 'dengue' or 'covid') and I'll fetch official guidance for you.")
        return str(resp)

    disease = parse_disease_from_text(body)
    region = None
    m_reg = re.search(r"in ([A-Za-z\s,]+)$", body, re.I)
    if m_reg:
        region = m_reg.group(1).strip()

    if not disease:
        resp.message("Sorry, I couldn't identify the disease. Please send a single disease name like 'dengue' or 'covid' (you can also say 'dengue in Kolkata').")
        return str(resp)

    try:
        result = query_sources(disease, region)
    except Exception as e:
        print("Error querying sources from WhatsApp webhook:", e)
        resp.message("Sorry, an internal error occurred while fetching guidance. Try again in a moment.")
        return str(resp)

    if not result.get("hits"):
        try:
            seed = globals().get("SEED_DB", {})
            if seed and seed.get(disease.lower()):
                out = seed.get(disease.lower())
                msg = format_whatsapp_reply_from_result(out, disease)
                resp.message(msg)
                return str(resp)
        except Exception:
            pass
        resp.message(f"Sorry — no official guidance found for '{disease}'. Try another disease or check spelling.")
        return str(resp)

    best = sorted(result["hits"], key=lambda x: x.get("confidence", 0), reverse=True)[0]
    msg_text = format_whatsapp_reply_from_result(best, disease, region)
    resp.message(msg_text)
    return str(resp)

def format_whatsapp_reply_from_result(hit, disease, region=None, max_chars=1500):
    lines = []
    title = hit.get("title") or disease.title()
    lines.append(f"{title}\n")
    summary = (hit.get("summary") or "").strip()
    if summary:
        first = summary.split("\n\n")[0].strip()
        if len(first) > 600:
            first = first[:600].rsplit(" ", 1)[0] + "…"
        lines.append(first)
    dos = hit.get("dos", [])[:3]
    donts = hit.get("donts", [])[:3]
    if dos:
        lines.append("\nDo's:")
        for d in dos:
            lines.append(f"• {d}")
    if donts:
        lines.append("\nDon'ts:")
        for d in donts:
            lines.append(f"• {d}")
    src = hit.get("source_url") or hit.get("source_name") or ""
    if src:
        lines.append(f"\nSource: {src}")
    lines.append("\nTo receive alerts for this disease in your area, reply: SUBSCRIBE " + disease.upper())
    return "\n".join(lines)

# --- Simple subscription storage (demo only) ---
SUBSCRIPTIONS = {}  # key: whatsapp_number -> list of {disease, region}

@app.route("/whatsapp_webhook_commands", methods=["POST"])
def whatsapp_commands_webhook():
    from_number = request.form.get("From")
    body = (request.form.get("Body") or "").strip()
    resp = MessagingResponse()
    b_up = body.strip().upper()
    if b_up.startswith("SUBSCRIBE"):
        parts = body.split()
        if len(parts) < 2:
            resp.message("To subscribe: SUBSCRIBE dengue [in <Region>]")
            return str(resp)
        disease = parts[1].lower()
        region = None
        m = re.search(r"in (.+)$", body, re.I)
        if m:
            region = m.group(1).strip()
        SUBSCRIPTIONS.setdefault(from_number, []).append({"disease": disease, "region": region})
        resp.message(f"Subscribed {from_number} to alerts for '{disease}'" + (f" in {region}" if region else ""))
        return str(resp)
    if b_up.startswith("UNSUBSCRIBE"):
        parts = body.split()
        if len(parts) == 1:
            SUBSCRIPTIONS.pop(from_number, None)
            resp.message("You have been unsubscribed from all alerts.")
            return str(resp)
        disease = parts[1].lower()
        subs = SUBSCRIPTIONS.get(from_number, [])
        subs = [s for s in subs if s.get("disease") != disease]
        SUBSCRIPTIONS[from_number] = subs
        resp.message(f"Unsubscribed from {disease} alerts.")
        return str(resp)
    return whatsapp_webhook()

def send_whatsapp_alert(to_whatsapp_number, message_text):
    """
    to_whatsapp_number should be in format 'whatsapp:+919999999999'
    message_text is the plain text message
    """
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        print("Twilio credentials missing.")
        return None
    if not twilio_client:
        print("Twilio client not initialized.")
        return None
    try:
        msg = twilio_client.messages.create(
            body=message_text,
            from_=TWILIO_WHATSAPP_FROM,
            to=to_whatsapp_number
        )
        return msg.sid
    except Exception as e:
        print("Error sending WhatsApp message:", e)
        return None

# ---- Run ----
if __name__ == "__main__":
    # debug=True only for development — remove in production
    app.run(debug=True, host="0.0.0.0", port=5000)
