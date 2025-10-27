import os
import io
import re
import json
import time
import pdfplumber
import docx
import nltk
import requests
import streamlit as st
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional OpenAI (only used if OPENAI_API_KEY is set)
try:
    from openai import OpenAI
    _OPENAI_IMPORTED = True
except Exception:
    _OPENAI_IMPORTED = False

# ---------- NLTK safe init ----------
try:
    _ = nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt")
    except Exception:
        pass

# ---------- Simple text utils ----------
def read_pdf(file_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text.append(t)
    return "\n".join(text)

def read_docx(file_bytes: bytes) -> str:
    f = io.BytesIO(file_bytes)
    doc = docx.Document(f)
    return "\n".join(p.text for p in doc.paragraphs)

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "")
    return t.strip()

def naive_sentence_split(t: str) -> List[str]:
    t = t or ""
    try:
        return nltk.sent_tokenize(t)
    except Exception:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]

# ---------- Skill extraction (lightweight keywords + regex) ----------
DEFAULT_SKILL_HINTS = [
    # General
    "python","java","c++","c","sql","mongodb","mysql","postgresql",
    "pandas","numpy","matplotlib","scikit-learn","tensorflow","pytorch",
    "nlp","cv","deep learning","machine learning","supervised","unsupervised",
    "transformers","rag","prompt engineering","fine-tuning","embedding",
    "langchain","oracle 23ai","oci","oci genai","vector search","llm","docker",
    "kubernetes","linux","git","rest api","streamlit","flask","react",
    # Cloud/OCI
    "oci object storage","oci data science","oci generative ai","oci language",
    "oci vision","oci speech","document understanding","select ai","supercluster"
]

def extract_skills(text: str, extra_hints: List[str] = None) -> List[str]:
    t = (text or "").lower()
    hints = set([h.lower() for h in (extra_hints or [])] + DEFAULT_SKILL_HINTS)
    found = []
    for h in hints:
        # match whole words, tolerate punctuation
        pattern = r"(?<!\w)"+re.escape(h)+r"(?!\w)"
        if re.search(pattern, t):
            found.append(h)
    found = sorted(set(found))
    # Title-case only for plain words, keep acronyms lower if they include spaces
    pretty = []
    for s in found:
        pretty.append(s if " " in s else s)
    return pretty

# ---------- LLM availability checks ----------
def oci_llm_available() -> bool:
    keys = [
        "OCI_GENAI_ENDPOINT","OCI_TENANCY_OCID","OCI_USER_OCID",
        "OCI_KEY_FINGERPRINT","OCI_KEY_FILE_PATH","OCI_COMPARTMENT_OCID","OCI_MODEL_ID"
    ]
    return all(os.getenv(k) for k in keys)

def openai_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and _OPENAI_IMPORTED

# ---------- LLM: OCI GenAI minimal wrapper ----------
def oci_sign_request(method: str, url: str, body: str = "") -> dict:
    """
    Placeholder headers. In production, use OCI SDK (RequestSigner) or a gateway.
    This will work only if your endpoint is fronted by something that doesn't require signed headers.
    Adjust for your lab setup if needed.
    """
    return {"Content-Type": "application/json"}

def call_oci_llm(prompt: str, max_tokens: int = 300, temperature: float = 0.2) -> str:
    endpoint = os.getenv("OCI_GENAI_ENDPOINT", "").rstrip("/")
    model_id = os.getenv("OCI_MODEL_ID", "cohere.command")
    if not endpoint:
        return ""
    headers = oci_sign_request("POST", endpoint, "")
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        # Adjust path to your service route if different
        url = f"{endpoint}/v1/chat/completions"
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return content or ""
        return ""
    except Exception:
        return ""

# ---------- LLM: OpenAI wrapper (optional) ----------
def call_openai_llm(prompt: str, max_tokens: int = 300, temperature: float = 0.2) -> str:
    if not openai_available():
        return ""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        chat = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return (chat.choices[0].message.content or "").strip()
    except Exception:
        return ""

# ---------- Fallback summariser ----------
def fallback_summary(text: str, lines: int = 4) -> str:
    # Extractive fallback: pick N informative sentences by length (quick & dirty)
    sents = naive_sentence_split(text)[:60]
    ranked = sorted(sents, key=lambda s: len(s), reverse=True)
    pick = ranked[:lines]
    return " ".join(pick).strip()

# ---------- Similarity & scoring ----------
def job_fit_score(resume_text: str, jd_text: str) -> Tuple[float, List[str]]:
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform([resume_text or "", jd_text or ""])
    sim = cosine_similarity(X[0], X[1])[0, 0]
    return float(sim), list(vec.get_feature_names_out())

def skill_gap(resume_skills: List[str], jd_skills: List[str]) -> List[str]:
    rs = set([s.lower() for s in (resume_skills or [])])
    js = set([s.lower() for s in (jd_skills or [])])
    gaps = sorted(list(js - rs))
    return [g for g in gaps]

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Resume Summariser & Job-Fit Analyzer", page_icon="🤖", layout="centered")
st.title("🤖 AI Resume Summariser & Job-Fit Analyzer")

st.markdown("""
Upload your **resume** and provide a **Job Description (JD)**.  
This app will:
- Extract **skills & entities**
- Create a **4-line professional summary** (OCI GenAI → or OpenAI → or Local fallback)
- Compute a **Job-Fit %** using TF-IDF cosine similarity
- Show **skill gaps** and **rewrite tips**
""")

with st.expander("⚙️ LLM Status"):
    st.write(f"OCI GenAI available: **{oci_llm_available()}**")
    st.write(f"OpenAI available: **{openai_available()}** (set `OPENAI_API_KEY` to enable)")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
with col2:
    jd_mode = st.radio("Job Description Input", ["Paste JD", "Upload JD File"], index=0)

jd_text = ""
if jd_mode == "Paste JD":
    jd_text = st.text_area("Paste Job Description here", height=200, placeholder="Paste JD...")
else:
    jd_file = st.file_uploader("Upload JD (PDF/DOCX/TXT)", type=["pdf","docx","txt"], key="jd")
    if jd_file:
        b = jd_file.read()
        if jd_file.type == "application/pdf" or jd_file.name.lower().endswith(".pdf"):
            jd_text = clean_text(read_pdf(b))
        elif jd_file.name.lower().endswith(".docx"):
            jd_text = clean_text(read_docx(b))
        else:
            jd_text = clean_text(b.decode("utf-8", errors="ignore"))

# ---------- Analyze action ----------
if st.button("Analyze"):
    if not resume_file:
        st.error("Please upload a resume.")
        st.stop()

    # Read resume
    rb = resume_file.read()
    if resume_file.type == "application/pdf" or resume_file.name.lower().endswith(".pdf"):
        resume_text = clean_text(read_pdf(rb))
    elif resume_file.name.lower().endswith(".docx"):
        resume_text = clean_text(read_docx(rb))
    else:
        resume_text = clean_text(rb.decode("utf-8", errors="ignore"))

    if not jd_text.strip():
        st.error("Please provide a Job Description (paste or upload).")
        st.stop()

    # --- Skills ---
    with st.spinner("Extracting skills..."):
        resume_sk = extract_skills(resume_text)
        jd_sk = extract_skills(jd_text)

    st.subheader("📌 Extracted Skills")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**From Resume**")
        st.write(", ".join(resume_sk) if resume_sk else "—")
    with c2:
        st.markdown("**From JD**")
        st.write(", ".join(jd_sk) if jd_sk else "—")

    # --- Summary ---
    with st.spinner("Generating professional summary..."):
        summary_prompt = summary_prompt = f"""
You are a helpful assistant generating a crisp professional summary for a resume.

CONTEXT (RESUME):
{resume_text[:8000]}

TASK:
Write a **4-line professional summary** tailored to software/data/ML roles.
- Be concise, specific, and action/impact-oriented.
- Prefer measurable outcomes (%, latency improvements, accuracy gains).
- Avoid buzzwords or unverifiable claims.
"""
