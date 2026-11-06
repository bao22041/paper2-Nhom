import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


class AppConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent
    ENV_FILE = BASE_DIR / ".env"
    DATA_DIR = BASE_DIR / "data"
    PDF_DIR = DATA_DIR / "pdfs"
    EXPORT_DIR = DATA_DIR / "exports"
    DB_PATH = DATA_DIR / "scientific_rag.db"

    MODEL_NAME = "Thiên Bảo NotebookLM Engine v7.0"
    DEFAULT_EVIDENCE_K = 8
    DEFAULT_CHUNK_SIZE = 1400
    DEFAULT_CHUNK_OVERLAP = 200

    OFFICIAL_TLDS: Dict[str, Dict[str, object]] = {
        ".gov.vn": {"score": 99, "badge": "🏛️ Cổng Chính phủ (.gov.vn)"},
        ".edu.vn": {"score": 98, "badge": "🎓 Đại học Việt Nam (.edu.vn)"},
        ".ac.vn":  {"score": 98, "badge": "🔬 Viện Nghiên cứu (.ac.vn)"},
        ".gov":    {"score": 99, "badge": "🏛️ Chính phủ Quốc tế (.gov)"},
        ".edu":    {"score": 98, "badge": "🎓 Đại học Quốc tế (.edu)"},
        ".ac.uk":  {"score": 98, "badge": "🎓 Đại học Anh (.ac.uk)"},
    }

    VERIFIED_DOMAINS: Dict[str, Dict[str, object]] = {
        "studocu.com":         {"score": 96, "badge": "📘 Studocu Đại học"},
        "slideshare.net":      {"score": 95, "badge": "📙 SlideShare Bài giảng"},
        "scribd.com":          {"score": 95, "badge": "📑 Scribd Học thuật"},
        "academia.edu":        {"score": 96, "badge": "🎓 Academia.edu"},
        "viblo.asia":          {"score": 96, "badge": "💻 Viblo CNTT"},
        "geeksforgeeks.org":   {"score": 96, "badge": "💻 GeeksforGeeks"},
        "freecodecamp.org":    {"score": 96, "badge": "💻 freeCodeCamp"},
        "kaggle.com":          {"score": 97, "badge": "📊 Kaggle AI/Data"},
        "huggingface.co":      {"score": 97, "badge": "🤗 HuggingFace AI"},
        "vjol.info.vn":        {"score": 99, "badge": "🇻🇳 Tạp chí Khoa học VN (VJOL)"},
        "luanvan.net.vn":      {"score": 94, "badge": "📑 Thư viện Luận văn VN"},
        "luanvan.co":          {"score": 94, "badge": "📑 Kho Luận văn"},
        "doc.edu.vn":          {"score": 94, "badge": "📘 Doc.edu.vn"},
        "tailieu.vn":          {"score": 92, "badge": "📂 TaiLieu.vn"},
        "vietjack.com":        {"score": 93, "badge": "📖 VietJack"},
        "vndoc.com":           {"score": 93, "badge": "📖 VnDoc"},
        "arxiv.org":           {"score": 98, "badge": "🔬 Cornell arXiv"},
        "semanticscholar.org": {"score": 98, "badge": "🔬 Semantic Scholar"},
        "ieee.org":            {"score": 99, "badge": "🔬 IEEE Xplore"},
        "springer.com":        {"score": 98, "badge": "🔬 Springer Nature"},
        "sciencedirect.com":   {"score": 99, "badge": "🔬 ScienceDirect"},
        "ncbi.nlm.nih.gov":    {"score": 99, "badge": "🏥 NCBI / PubMed"},
        "researchgate.net":    {"score": 96, "badge": "🔬 ResearchGate"},
        "wikipedia.org":       {"score": 93, "badge": "📖 Wikipedia"},
    }

    BLOCKED_DOMAIN_KEYWORDS: List[str] = [
        "facebook.com", "fb.com", "tiktok.com", "instagram.com",
        "shopee.", "lazada.", "tiki.", "bet", "casino", "kubet", "nhacai", "xoso",
    ]

    @classmethod
    def ensure_dirs(cls):
        cls.PDF_DIR.mkdir(parents=True, exist_ok=True)
        cls.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        cls.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_api_key(cls) -> str:
        load_dotenv(cls.ENV_FILE, override=True)
        return os.getenv("GEMINI_API_KEY", "").strip()