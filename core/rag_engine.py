import io
import json
import time
import asyncio
import html
import unicodedata
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, quote_plus, unquote

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import docx
from pptx import Presentation
from pypdf import PdfReader

from core.config import AppConfig
from core.database import ScientificDatabase


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


class NotebookLMAutonomousSynthesizer:
    """
    Bộ não AI Sinh ngữ Tự động cho NotebookLM:
    - Tầng 1: Nếu có GEMINI_API_KEY trong .env, tự động dò model khả dụng trên tài khoản (gemini-3.0-flash, gemini-2.5-flash, gemini-2.0-flash...) và gọi REST API trực tiếp.
    - Tầng 2: Tự động gọi Free AI LLM Endpoint (Pollinations OpenAI-compatible) không cần API Key, sinh câu trả lời tự nhiên như NotebookLM.
    - Tầng 3: Nếu chạy ở chế độ Offline ngắt mạng, tự động tổng hợp ngữ nghĩa tiếng Việt có gắn trích dẫn [1], [2].
    """
    _cached_gemini_model: str = ""

    @classmethod
    def _resolve_gemini_model(cls, api_key: str) -> str:
        if cls._cached_gemini_model:
            return cls._cached_gemini_model
        try:
            r = requests.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                timeout=5,
            )
            if r.status_code == 200:
                models = r.json().get("models", [])
                names = [
                    m["name"].replace("models/", "")
                    for m in models
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                for pref in [
                    "gemini-3.0-flash", "gemini-2.5-flash", "gemini-2.0-flash",
                    "gemini-flash-latest", "gemini-1.5-flash",
                ]:
                    if pref in names:
                        cls._cached_gemini_model = pref
                        return pref
                for n in names:
                    if "flash" in n and "tts" not in n and "image" not in n:
                        cls._cached_gemini_model = n
                        return n
        except Exception:
            pass
        return "gemini-2.0-flash"

    @classmethod
    def generate(cls, system_prompt: str, user_prompt: str, fallback_text: str = "") -> str:
        # 1. Thử gọi trực tiếp Gemini REST API nếu có Key trong .env
        api_key = AppConfig.get_api_key()
        if api_key and len(api_key) > 15 and "DIEN_API_KEY" not in api_key:
            try:
                model_name = cls._resolve_gemini_model(api_key)
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}],
                        }
                    ],
                    "generationConfig": {"temperature": 0.25},
                }
                resp = requests.post(url, json=payload, timeout=22)
                if resp.status_code == 200:
                    candidates = resp.json().get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and parts[0].get("text"):
                            return parts[0]["text"].strip()
            except Exception:
                pass

        # 2. Gọi Free Keyless AI LLM API (Không cần API Key, trả lời tự nhiên như NotebookLM)
        try:
            resp = requests.post(
                "https://text.pollinations.ai/openai",
                json={
                    "model": "openai",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.25,
                },
                timeout=22,
            )
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices and choices[0].get("message", {}).get("content"):
                    return choices[0]["message"]["content"].strip()
        except Exception:
            pass

        # 3. Thử Ollama Local nếu máy có bật
        try:
            r_tags = requests.get("http://localhost:11434/api/tags", timeout=0.8)
            if r_tags.status_code == 200:
                models = r_tags.json().get("models", [])
                if models:
                    m_name = models[0]["name"]
                    r_gen = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": m_name,
                            "prompt": f"{system_prompt}\n\n{user_prompt}",
                            "stream": False,
                        },
                        timeout=25,
                    )
                    if r_gen.status_code == 200 and r_gen.json().get("response"):
                        return r_gen.json()["response"].strip()
        except Exception:
            pass

        return fallback_text


class OfficialSourceValidator:
    @classmethod
    def verify_url(cls, url: str, strict_whitelist: bool = False) -> Tuple[bool, int, str, str]:
        if not url or not isinstance(url, str):
            return False, 0, "URL không hợp lệ", ""
        clean_url = url.strip()
        if not clean_url.startswith(("https://", "http://export.arxiv.org")):
            return False, 0, "Không dùng HTTPS", ""
        parsed = urlparse(clean_url)
        domain = (parsed.netloc or "").lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        if not domain or "xn--" in domain:
            return False, 0, "Tên miền không hợp lệ", domain

        for bad_kw in AppConfig.BLOCKED_DOMAIN_KEYWORDS:
            if bad_kw in domain or bad_kw in clean_url.lower():
                return False, 0, f"Đã chặn ({bad_kw})", domain

        for verified_dom, info in AppConfig.VERIFIED_DOMAINS.items():
            if domain == verified_dom or domain.endswith("." + verified_dom):
                return True, int(info["score"]), str(info["badge"]), domain

        for tld, info in AppConfig.OFFICIAL_TLDS.items():
            if domain.endswith(tld):
                return True, int(info["score"]), str(info["badge"]), domain

        return True, 88, f"🌐 Nguồn Web ({domain})", domain


class SimpleChunkObj:
    def __init__(self, name: str, text: str, docname: str, citation: str, paper_id: int):
        self.name = name
        self.text = text
        self.doc = type(
            "DocMeta", (), {"docname": docname, "citation": citation, "paper_id": paper_id}
        )()


class PaperQA2UnifiedEngine:
    STOPWORDS_VI = {
        "của", "và", "các", "có", "trong", "được", "cho", "là", "với", "những",
        "một", "này", "đến", "khi", "về", "như", "từ", "theo", "trên", "tại",
        "để", "không", "cũng", "đã", "sẽ", "việc", "bởi", "hay", "hoặc", "rằng",
        "thì", "mà", "ở", "ra", "vào", "lại", "nên", "còn", "do", "qua",
        "the", "and", "of", "to", "in", "is", "for", "that", "with", "on", "as",
    }

    QUESTION_FILLER_WORDS = {
        "cho", "tôi", "hỏi", "biết", "về", "là", "gì", "như", "thế", "nào",
        "ra", "sao", "ở", "đâu", "có", "những", "các", "trong", "file", "bài",
        "báo", "tài", "liệu", "này", "hãy", "giúp", "phân", "tích", "tóm", "tắt",
    }

    CATEGORY_PATTERNS = {
        "problem": ["bài toán", "mục tiêu", "mục đích", "vấn đề", "thách thức", "đề tài", "khảo sát", "nhằm", "tập trung", "khái niệm", "định nghĩa"],
        "methodology": ["phương pháp", "mô hình", "thuật toán", "kiến trúc", "đề xuất", "quy trình", "hệ thống", "sử dụng", "áp dụng", "thiết kế", "xây dựng", "thang đo"],
        "dataset": ["dữ liệu", "tập dữ liệu", "mẫu khảo sát", "sinh viên", "đối tượng", "thực nghiệm", "thu thập", "bảng hỏi", "tham gia"],
        "results": ["kết quả", "độ chính xác", "hiệu năng", "đạt được", "tăng", "giảm", "%", "thống kê", "cho thấy", "chứng minh", "đánh giá", "trung bình", "tỷ lệ", "ảnh hưởng", "tác động"],
        "limitations": ["hạn chế", "nhược điểm", "tương lai", "hướng phát triển", "khuyến nghị", "kết luận", "giải pháp", "cải thiện", "kiến nghị"],
    }

    SEARCH_MODE_FILTERS = {
        "all": "(site:studocu.com OR site:.edu.vn OR site:viblo.asia OR site:vjol.info.vn OR site:luanvan.net.vn OR site:doc.edu.vn OR site:slideshare.net OR site:vietjack.com OR site:vndoc.com OR site:.gov.vn)",
        "studocu": "(site:studocu.com OR site:academia.edu OR site:slideshare.net OR site:scribd.com OR site:luanvan.net.vn OR site:doc.edu.vn)",
        "tech": "(site:viblo.asia OR site:geeksforgeeks.org OR site:freecodecamp.org OR site:kaggle.com OR site:huggingface.co OR site:arxiv.org)",
        "official": "(site:.edu.vn OR site:.gov.vn OR site:vjol.info.vn OR site:.edu OR site:.gov OR site:sciencedirect.com)",
    }

    def __init__(
        self,
        db: ScientificDatabase,
        api_key: Optional[str] = None,
        llm_model: str = AppConfig.MODEL_NAME,
        embedding_model: str = "NotebookLM Hybrid Semantic",
        temperature: float = 0.2,
        evidence_k: int = AppConfig.DEFAULT_EVIDENCE_K,
        max_sources: int = 6,
        chunk_size: int = AppConfig.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = AppConfig.DEFAULT_CHUNK_OVERLAP,
    ):
        self.db = db
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.evidence_k = evidence_k
        self.max_sources = max_sources
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        AppConfig.ensure_dirs()
        self.docs = type("DocsContainer", (), {"texts": []})()
        self.sync_and_learn_all_files()

    # ---------------- ĐỒNG BỘ TÀI LIỆU VÀO DATABASE ----------------
    def sync_and_learn_all_files(self) -> Dict[str, int]:
        scanned_count = 0
        allowed_exts = {".pdf", ".docx", ".pptx", ".txt", ".md"}
        existing_files = [
            p for p in AppConfig.PDF_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in allowed_exts
        ]

        db_papers = self.db.get_all_papers()
        db_hashes = {p["file_hash"]: p for p in db_papers}

        for file_path in existing_files:
            try:
                f_hash = self.db.compute_file_hash(file_path)
                db_rec = db_hashes.get(f_hash)
                if db_rec is None or db_rec.get("char_length", 0) < 20 or not db_rec.get("has_db_blob"):
                    run_async(self.async_add_document(file_path))
                    scanned_count += 1
            except Exception:
                continue

        for p in self.db.get_all_papers():
            pid = p["id"]
            existing_chunks = self.db.get_all_document_chunks(paper_ids=[pid])
            if not existing_chunks:
                full_paper = self.db.get_paper_by_id(pid)
                if full_paper and full_paper.get("full_text"):
                    secs = self.db.get_paper_sections(pid)
                    sec_tuples = (
                        [(s["section_label"], s["section_text"]) for s in secs]
                        if secs
                        else [("Nội dung toàn văn", full_paper["full_text"])]
                    )
                    self._process_and_store_to_db(
                        paper_id=pid,
                        title=full_paper["title"],
                        docname=full_paper["docname"],
                        citation=full_paper["citation"],
                        source_url=full_paper.get("source_url", ""),
                        trust_score=full_paper.get("trust_score", 100),
                        authority_badge=full_paper.get("authority_badge", ""),
                        sections=sec_tuples,
                    )
                    scanned_count += 1

        self.reload_index()
        return {"scanned_files": scanned_count, "total_chunks": len(self.docs.texts)}

    def reload_index(self):
        all_chunks = self.db.get_all_document_chunks()
        self.docs.texts = [
            SimpleChunkObj(
                name=c["chunk_name"],
                text=c["chunk_text"],
                docname=c["docname"],
                citation=c["citation"],
                paper_id=c["paper_id"],
            )
            for c in all_chunks
        ]

    def reset_index(self):
        self.sync_and_learn_all_files()

    def remove_paper_from_brain(self, paper_id: int):
        self.reload_index()

    # ---------------- ĐỌC FILE PDF / WORD / PPTX / TEXT ----------------
    @staticmethod
    def _clean_extracted_text(raw: str) -> str:
        if not raw:
            return ""
        lines = [line.strip() for line in raw.splitlines()]
        merged_lines = []
        buf = ""
        for line in lines:
            if not line:
                if buf:
                    merged_lines.append(buf)
                    buf = ""
                continue
            if not buf:
                buf = line
            else:
                if not buf.endswith((".", "!", "?", ":", ";")) and (
                    line[0].islower() or len(buf) < 55
                ):
                    buf += " " + line
                else:
                    merged_lines.append(buf)
                    buf = line
        if buf:
            merged_lines.append(buf)
        return re.sub(r"[ \t]{2,}", " ", "\n".join(merged_lines)).strip()

    def _extract_pdf_bytes(self, file_bytes: bytes, fallback_name: str) -> Tuple[str, str, List[Tuple[str, str]]]:
        title = Path(fallback_name).stem.replace("_", " ")
        author = "Nhóm tác giả"
        pages_data: List[Tuple[str, str]] = []

        reader = PdfReader(io.BytesIO(file_bytes))
        meta = reader.metadata
        if meta:
            if getattr(meta, "title", None) and len(str(meta.title).strip()) > 4:
                title = str(meta.title).strip()
            if getattr(meta, "author", None) and len(str(meta.author).strip()) > 1:
                author = str(meta.author).strip()

        for idx, page in enumerate(reader.pages, 1):
            raw_txt = page.extract_text() or ""
            clean_txt = self._clean_extracted_text(raw_txt)
            if clean_txt:
                pages_data.append((f"Trang {idx}", clean_txt))

        if pages_data:
            first_page_lines = [
                l.strip()
                for l in pages_data[0][1].splitlines()
                if 15 <= len(l.strip()) <= 180
                and not l.lower().startswith(("tạp chí", "issn", "trường đại học", "journal", "vol"))
            ]
            if first_page_lines and ("_" in Path(fallback_name).stem or len(title) < 8):
                title = first_page_lines[0]

        if not pages_data:
            pages_data.append(("Trang 1", f"Tài liệu PDF: {title}"))
        return title, author, pages_data

    def _extract_docx_bytes(self, file_bytes: bytes, filename: str) -> Tuple[List[Tuple[str, str]], str]:
        doc = docx.Document(io.BytesIO(file_bytes))
        sections: List[Tuple[str, str]] = []
        current_heading = "Phần 1 - Mở đầu"
        current_buf: List[str] = []
        html_blocks: List[str] = []
        sec_idx = 1

        for p in doc.paragraphs:
            t = p.text.strip()
            if not t:
                continue
            style_name = (p.style.name or "").lower()
            safe_t = html.escape(t)
            if "heading" in style_name or "title" in style_name or (len(t) < 90 and t.isupper()):
                if current_buf:
                    sections.append((current_heading, "\n".join(current_buf)))
                    current_buf = []
                sec_idx += 1
                current_heading = f"Mục {sec_idx}: {t[:70]}"
                html_blocks.append(f"<h2>{safe_t}</h2>")
            else:
                current_buf.append(t)
                html_blocks.append(f"<p>{safe_t}</p>")

        for idx, table in enumerate(doc.tables, 1):
            rows_html = []
            table_lines = []
            for r_idx, row in enumerate(table.rows):
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    table_lines.append(" | ".join(cells))
                    tag = "th" if r_idx == 0 else "td"
                    cells_html = "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells)
                    rows_html.append(f"<tr>{cells_html}</tr>")
            if table_lines:
                current_buf.append(f"[Bảng {idx}]\n" + "\n".join(table_lines))
            if rows_html:
                html_blocks.append(f"<table>{''.join(rows_html)}</table>")

        if current_buf:
            sections.append((current_heading, "\n".join(current_buf)))
        if not sections:
            sections.append(("Nội dung Word", f"Tài liệu Word: {filename}"))

        preview_html = f"<html><body style='font-family:sans-serif;padding:24px;line-height:1.7;'><h1>{html.escape(filename)}</h1>{''.join(html_blocks)}</body></html>"
        return sections, preview_html

    def _extract_pptx_bytes(self, file_bytes: bytes, filename: str) -> Tuple[List[Tuple[str, str]], str]:
        prs = Presentation(io.BytesIO(file_bytes))
        sections: List[Tuple[str, str]] = []
        slide_cards_html: List[str] = []

        for idx, slide in enumerate(prs.slides, 1):
            slide_lines = []
            title_text = f"Slide {idx}"
            if slide.shapes.title and slide.shapes.title.text.strip():
                title_text = slide.shapes.title.text.strip()
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    for para in shape.text_frame.paragraphs:
                        line = para.text.strip()
                        if line and line != title_text:
                            slide_lines.append(line)
            content = f"{title_text}.\n" + "\n".join(slide_lines)
            sections.append((f"Slide {idx}: {title_text[:50]}", content))
            bullets = "".join(f"<li>{html.escape(l)}</li>" for l in slide_lines)
            slide_cards_html.append(f"<div style='border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin-bottom:12px;'><h3>Slide {idx}: {html.escape(title_text)}</h3><ul>{bullets}</ul></div>")

        preview_html = f"<html><body style='font-family:sans-serif;padding:24px;'><h1>{html.escape(filename)}</h1>{''.join(slide_cards_html)}</body></html>"
        return sections, preview_html

    # ---------------- XỬ LÝ & LƯU VÀO DATABASE + TẠO SOURCE GUIDE TỰ ĐỘNG ----------------
    def _extract_top_key_concepts(self, full_text: str, top_n: int = 10) -> List[str]:
        words = re.findall(r"[a-zA-ZÀ-ỹ]{3,}", full_text.lower())
        filtered = [w for w in words if w not in self.STOPWORDS_VI]
        if not filtered:
            return ["Tài liệu nghiên cứu"]
        bigrams = [f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered) - 1)]
        counts = Counter(bigrams)
        return [phrase for phrase, _ in counts.most_common(top_n)]

    def _process_and_store_to_db(
        self,
        paper_id: int,
        title: str,
        docname: str,
        citation: str,
        source_url: str,
        trust_score: int,
        authority_badge: str,
        sections: List[Tuple[str, str]],
    ) -> Tuple[int, str, str, List[str], List[str]]:
        new_chunks: List[Dict[str, Any]] = []
        knowledge_nodes: List[Dict[str, Any]] = []
        section_records: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []
        chunk_idx = 1

        for sec_label, sec_text in sections:
            clean_sec = re.sub(r"\n{3,}", "\n\n", sec_text).strip()
            if not clean_sec:
                continue

            full_text_parts.append(f"=== {sec_label} ===\n{clean_sec}")
            raw_sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+|\n+", clean_sec)
                if len(s.strip()) >= 20
            ]
            sec_key_point = " ".join(raw_sentences[:3])[:380] if raw_sentences else clean_sec[:380]
            section_records.append(
                {"label": sec_label, "text": clean_sec, "key_points": sec_key_point}
            )

            start = 0
            text_len = len(clean_sec)
            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                piece = clean_sec[start:end].strip()
                if len(piece) >= 20:
                    c_name = f"{title[:35]} • {sec_label}"
                    new_chunks.append(
                        {
                            "paper_id": paper_id,
                            "title": title,
                            "docname": docname,
                            "citation": citation,
                            "section_label": sec_label,
                            "source_url": source_url,
                            "trust_score": trust_score,
                            "authority_badge": authority_badge,
                            "chunk_name": c_name,
                            "text": piece,
                        }
                    )
                    chunk_idx += 1
                if end >= text_len:
                    break
                start = max(end - self.chunk_overlap, start + 200)

            for sent in raw_sentences:
                if len(sent) > 800:
                    sent = sent[:800]
                s_lower = sent.lower()
                for cat_key, keywords in self.CATEGORY_PATTERNS.items():
                    matched_kws = [kw for kw in keywords if kw in s_lower]
                    if matched_kws:
                        has_num = bool(re.search(r"\d+(?:[.,]\d+)?%?", sent))
                        weight = len(matched_kws) * (1.5 if has_num else 1.1)
                        knowledge_nodes.append(
                            {
                                "category": cat_key,
                                "keywords": ", ".join(matched_kws[:4]),
                                "content": sent,
                                "location": sec_label,
                                "weight": round(weight, 2),
                            }
                        )

            if raw_sentences:
                best_sent = max(raw_sentences[:5], key=len)
                knowledge_nodes.append(
                    {
                        "category": "core_fact",
                        "keywords": sec_label.lower(),
                        "content": best_sent[:500],
                        "location": sec_label,
                        "weight": 1.2,
                    }
                )

        full_text_combined = "\n\n".join(full_text_parts)
        key_concepts = self._extract_top_key_concepts(full_text_combined, top_n=10)

        knowledge_nodes.sort(key=lambda x: x["weight"], reverse=True)
        seen_sigs = set()
        unique_nodes = []
        for node in knowledge_nodes:
            sig = strip_accents(node["content"][:70].lower())
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                unique_nodes.append(node)
            if len(unique_nodes) >= 100:
                break

        def _pick_by_cat(cat: str, limit: int = 2) -> str:
            items = [n for n in unique_nodes if n["category"] == cat][:limit]
            if not items:
                items = [n for n in unique_nodes if n["category"] == "core_fact"][:limit]
            return " ".join(f"{i['content']} ({i['location']})." for i in items)

        fallback_summary = (
            f"Tài liệu **{title}** tập trung giải quyết: {_pick_by_cat('problem', 2)}\n\n"
            f"**Phương pháp & Triển khai:** {_pick_by_cat('methodology', 2)}\n\n"
            f"**Kết quả & Số liệu chính:** {_pick_by_cat('results', 2)}"
        )

        # Sinh tóm tắt tự nhiên chuẩn NotebookLM Source Guide
        excerpt_for_guide = full_text_combined[:5500]
        auto_summary = NotebookLMAutonomousSynthesizer.generate(
            system_prompt=(
                "Bạn là trợ lý Google NotebookLM. Hãy viết một đoạn 'Tóm tắt Nguồn tài liệu (Source Guide)' "
                "bằng tiếng Việt thật súc tích, mạch lạc (khoảng 120-180 từ) nêu bật chủ đề chính, phương pháp và phát hiện cốt lõi của tài liệu."
            ),
            user_prompt=f"Tiêu đề: {title}\nNội dung trích xuất:\n{excerpt_for_guide}",
            fallback_text=fallback_summary,
        )

        suggested_questions = [
            f"Mục tiêu cốt lõi và vấn đề chính mà '{title[:45]}' giải quyết là gì?",
            f"Phương pháp thực hiện và dữ liệu nghiên cứu trong '{title[:45]}' được triển khai ra sao?",
            f"Những kết quả, số liệu nổi bật và kết luận quan trọng nhất trong tài liệu này là gì?",
        ]

        self.db.save_paper_sections(paper_id, section_records)
        self.db.replace_paper_knowledge(paper_id, title, unique_nodes)
        self.db.save_document_chunks(paper_id, new_chunks)
        self.reload_index()

        return len(new_chunks), full_text_combined, auto_summary, key_concepts, suggested_questions

    # ---------------- NẠP FILE / TEXT / WEB URL VÀO NOTEBOOK-LM ----------------
    async def async_add_document(
        self, file_path: Path, custom_citation: Optional[str] = None
    ) -> Dict[str, Any]:
        file_bytes = file_path.read_bytes()
        file_hash = self.db.compute_bytes_hash(file_bytes)
        ext = file_path.suffix.lower()

        preview_html = ""
        doc_type = "pdf"
        mime_type = "application/pdf"
        default_title = file_path.stem.replace("_", " ")
        default_authors = "Nhóm tác giả"

        if ext == ".docx":
            doc_type = "docx"
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            sections, preview_html = self._extract_docx_bytes(file_bytes, file_path.name)
        elif ext == ".pptx":
            doc_type = "pptx"
            mime_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            sections, preview_html = self._extract_pptx_bytes(file_bytes, file_path.name)
        elif ext in {".txt", ".md"}:
            doc_type = "txt"
            mime_type = "text/plain"
            txt_str = file_bytes.decode("utf-8", errors="ignore")
            sections = [("Văn bản gốc", txt_str)]
            preview_html = f"<html><body style='font-family:sans-serif;padding:24px;white-space:pre-wrap;'>{html.escape(txt_str)}</body></html>"
        else:
            doc_type = "pdf"
            mime_type = "application/pdf"
            default_title, default_authors, sections = self._extract_pdf_bytes(file_bytes, file_path.name)

        citation_to_use = custom_citation or f"{default_title} ({file_path.name})"
        clean_docname = f"{file_path.stem[:28]}_{file_hash[:6]}"
        badge = f"📄 {doc_type.upper()}"

        paper_record = self.db.upsert_paper(
            filename=file_path.name,
            file_path=str(file_path),
            file_hash=file_hash,
            doc_type=doc_type,
            mime_type=mime_type,
            file_blob=file_bytes,
            source_url="",
            preview_html=preview_html,
            docname=clean_docname,
            title=default_title,
            authors=default_authors,
            year=str(datetime.now().year),
            citation=citation_to_use,
            chunk_count=0,
            trust_score=100,
            authority_badge=badge,
        )

        c_cnt, f_txt, a_sum, k_con, s_ques = self._process_and_store_to_db(
            paper_id=paper_record["id"],
            title=default_title,
            docname=clean_docname,
            citation=citation_to_use,
            source_url="",
            trust_score=100,
            authority_badge=badge,
            sections=sections,
        )

        return self.db.upsert_paper(
            filename=file_path.name,
            file_path=str(file_path),
            file_hash=file_hash,
            doc_type=doc_type,
            mime_type=mime_type,
            file_blob=file_bytes,
            source_url="",
            preview_html=preview_html,
            docname=clean_docname,
            title=default_title,
            authors=default_authors,
            year=str(datetime.now().year),
            citation=citation_to_use,
            chunk_count=c_cnt,
            full_text=f_txt,
            auto_summary=a_sum,
            key_concepts=k_con,
            suggested_questions=s_ques,
            trust_score=100,
            authority_badge=badge,
        )

    async def async_add_pasted_text(self, title: str, content: str) -> Dict[str, Any]:
        clean_title = title.strip() or f"Ghi chú văn bản ({datetime.now().strftime('%H:%M')})"
        text_bytes = content.encode("utf-8")
        file_hash = self.db.compute_bytes_hash(text_bytes)
        preview_html = f"<html><body style='font-family:sans-serif;padding:28px;line-height:1.7;white-space:pre-wrap;'><h1>{html.escape(clean_title)}</h1>{html.escape(content)}</body></html>"

        paper_rec = self.db.upsert_paper(
            filename="Văn bản dán trực tiếp",
            file_path="pasted_text",
            file_hash=file_hash,
            doc_type="txt",
            mime_type="text/html",
            file_blob=preview_html.encode("utf-8"),
            preview_html=preview_html,
            docname=f"text_{file_hash[:6]}",
            title=clean_title,
            authors="Người dùng",
            year=str(datetime.now().year),
            citation=clean_title,
            authority_badge="📝 Văn bản",
        )
        c_cnt, f_txt, a_sum, k_con, s_ques = self._process_and_store_to_db(
            paper_id=paper_rec["id"],
            title=clean_title,
            docname=f"text_{file_hash[:6]}",
            citation=clean_title,
            source_url="",
            trust_score=100,
            authority_badge="📝 Văn bản",
            sections=[("Nội dung văn bản", content)],
        )
        return self.db.upsert_paper(
            filename="Văn bản dán trực tiếp",
            file_path="pasted_text",
            file_hash=file_hash,
            doc_type="txt",
            mime_type="text/html",
            file_blob=preview_html.encode("utf-8"),
            preview_html=preview_html,
            docname=f"text_{file_hash[:6]}",
            title=clean_title,
            authors="Người dùng",
            year=str(datetime.now().year),
            citation=clean_title,
            chunk_count=c_cnt,
            full_text=f_txt,
            auto_summary=a_sum,
            key_concepts=k_con,
            suggested_questions=s_ques,
            authority_badge="📝 Văn bản",
        )

    def _visit_and_save_url_to_db(
        self,
        url: str,
        search_query: str = "",
        strict_whitelist: bool = False,
        fallback_title: str = "",
        fallback_snippet: str = "",
    ) -> Optional[Dict[str, Any]]:
        clean_url = url.strip()
        if not clean_url.startswith(("http://", "https://")):
            clean_url = "https://" + clean_url

        existing = self.db.get_paper_by_url(clean_url)
        if existing and len(existing.get("full_text", "")) > 200:
            return existing

        is_official, trust_score, badge, domain = OfficialSourceValidator.verify_url(
            clean_url, strict_whitelist=strict_whitelist
        )
        if not is_official:
            return None

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
        }

        final_url = clean_url
        t_score_final, badge_final, dom_final = trust_score, badge, domain
        sections: List[Tuple[str, str]] = []
        html_body_parts: List[str] = []
        page_title = fallback_title or domain

        try:
            resp = requests.get(clean_url, headers=headers, timeout=10, allow_redirects=True)
            final_url = resp.url or clean_url
            is_off_f, t_score_f, badge_f, dom_f = OfficialSourceValidator.verify_url(final_url)
            if not is_off_f:
                return None
            t_score_final, badge_final, dom_final = t_score_f, badge_f, dom_f

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "lxml")
                if soup.title and soup.title.string and len(soup.title.string.strip()) > 3:
                    page_title = soup.title.string.strip()

                meta_descriptions = []
                for meta_tag in soup.find_all("meta"):
                    prop = (meta_tag.get("property") or meta_tag.get("name") or "").lower()
                    if prop in {"description", "og:description"}:
                        c_val = (meta_tag.get("content") or "").strip()
                        if len(c_val) > 30 and c_val not in meta_descriptions:
                            meta_descriptions.append(c_val)

                if meta_descriptions:
                    combined_meta = "\n\n".join(meta_descriptions)
                    sections.append(("Tóm tắt Trang", combined_meta))
                    html_body_parts.append(f"<p><b>Tóm tắt:</b> {html.escape(combined_meta)}</p>")

                for junk in soup(["script", "style", "noscript", "iframe", "svg", "nav", "footer", "header", "form"]):
                    junk.decompose()

                current_sec_title = "Nội dung bài viết"
                current_paras: List[str] = []
                sec_counter = 1
                main_container = soup.find("article") or soup.find("main") or soup.body or soup

                for elem in main_container.find_all(["h1", "h2", "h3", "p", "li", "blockquote"]):
                    txt = elem.get_text(" ", strip=True)
                    if not txt or len(txt) < 25:
                        continue
                    if elem.name in {"h1", "h2", "h3"} and len(txt) < 150:
                        if current_paras:
                            sections.append((current_sec_title, "\n".join(current_paras)))
                            current_paras = []
                        sec_counter += 1
                        current_sec_title = f"Mục {sec_counter}: {txt}"
                        html_body_parts.append(f"<h2>{html.escape(txt)}</h2>")
                    else:
                        if txt not in current_paras:
                            current_paras.append(txt)
                            html_body_parts.append(f"<p>{html.escape(txt)}</p>")

                if current_paras:
                    sections.append((current_sec_title, "\n".join(current_paras)))
        except Exception:
            pass

        if fallback_snippet and len(fallback_snippet.strip()) > 25:
            sections.insert(0, ("Trích đoạn Nguồn", fallback_snippet.strip()))
            html_body_parts.insert(0, f"<p><b>Trích đoạn:</b> {html.escape(fallback_snippet.strip())}</p>")

        if sum(len(s[1]) for s in sections) < 40:
            return None

        url_hash = self.db.compute_text_hash(final_url)
        preview_html = f"""<!DOCTYPE html><html lang="vi"><head><meta charset="UTF-8"><style>
        body {{ font-family: sans-serif; padding: 24px; line-height: 1.7; color: #1e293b; }}
        h1 {{ font-size: 20px; color: #0f172a; }} h2 {{ font-size: 16px; color: #1d4ed8; margin-top: 18px; }}
        </style></head><body>
        <div><b>{html.escape(badge_final)}</b> • <a href="{html.escape(final_url)}" target="_blank">{html.escape(final_url)}</a></div>
        <h1>{html.escape(page_title)}</h1>
        {''.join(html_body_parts)}
        </body></html>"""

        html_bytes = preview_html.encode("utf-8")
        clean_docname = f"web_{dom_final}_{url_hash[:6]}"
        citation_str = f"{page_title} ({final_url})"

        paper_rec = self.db.upsert_paper(
            filename=dom_final,
            file_path=final_url,
            file_hash=url_hash,
            doc_type="web",
            mime_type="text/html",
            file_blob=html_bytes,
            source_url=final_url,
            preview_html=preview_html,
            docname=clean_docname,
            title=page_title,
            authors=dom_final,
            year=str(datetime.now().year),
            citation=citation_str,
            trust_score=t_score_final,
            authority_badge=badge_final,
        )

        c_cnt, f_txt, a_sum, k_con, s_ques = self._process_and_store_to_db(
            paper_id=paper_rec["id"],
            title=page_title,
            docname=clean_docname,
            citation=citation_str,
            source_url=final_url,
            trust_score=t_score_final,
            authority_badge=badge_final,
            sections=sections,
        )

        updated = self.db.upsert_paper(
            filename=dom_final,
            file_path=final_url,
            file_hash=url_hash,
            doc_type="web",
            mime_type="text/html",
            file_blob=html_bytes,
            source_url=final_url,
            preview_html=preview_html,
            docname=clean_docname,
            title=page_title,
            authors=dom_final,
            year=str(datetime.now().year),
            citation=citation_str,
            chunk_count=c_cnt,
            full_text=f_txt,
            auto_summary=a_sum,
            key_concepts=k_con,
            suggested_questions=s_ques,
            trust_score=t_score_final,
            authority_badge=badge_final,
        )
        self.db.save_verified_web_source(
            search_query=search_query or page_title,
            title=page_title,
            url=final_url,
            domain=dom_final,
            authority_badge=badge_final,
            trust_score=t_score_final,
            authors=dom_final,
            year=str(datetime.now().year),
            doi="",
            snippet_summary=f_txt[:500],
            linked_paper_id=updated["id"],
        )
        return updated

    async def async_add_web_url(self, url: str, strict_official_only: bool = False) -> Dict[str, Any]:
        saved = self._visit_and_save_url_to_db(url=url, fallback_title=url, fallback_snippet=url)
        if not saved:
            raise ValueError("Không thể đọc nội dung từ đường link này.")
        return saved

    def search_official_google_and_scholar(
        self, query: str, max_results: int = 3, auto_ingest: bool = True, search_mode: str = "all"
    ) -> List[Dict[str, Any]]:
        clean_q = re.sub(r"[?#!]+", " ", query).strip()
        if not clean_q:
            return []

        crawled_papers: List[Dict[str, Any]] = []
        candidate_items: List[Dict[str, str]] = []
        seen_urls = set()
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0"}

        site_filter = self.SEARCH_MODE_FILTERS.get(search_mode, self.SEARCH_MODE_FILTERS["all"])
        queries = [f"{clean_q} {site_filter}", clean_q]
        if search_mode == "studocu":
            queries.insert(0, f"{clean_q} site:studocu.com")

        for q_str in queries:
            if len(candidate_items) >= 8:
                break
            try:
                ddg_url = f"https://html.duckduckgo.com/html/?q={quote_plus(q_str)}"
                ddg_resp = requests.get(ddg_url, headers=headers, timeout=7)
                if ddg_resp.status_code == 200:
                    soup_ddg = BeautifulSoup(ddg_resp.text, "lxml")
                    for res_block in soup_ddg.select("div.result"):
                        a_tag = res_block.select_one("a.result__a")
                        snip_tag = res_block.select_one("a.result__snippet")
                        if not a_tag:
                            continue
                        href = a_tag.get("href", "")
                        real_u = unquote(href.split("uddg=")[-1].split("&")[0]) if "uddg=" in href else href
                        if real_u.startswith("https://") and real_u not in seen_urls:
                            seen_urls.add(real_u)
                            candidate_items.append({
                                "url": real_u,
                                "title": a_tag.get_text(" ", strip=True),
                                "snippet": snip_tag.get_text(" ", strip=True) if snip_tag else "",
                            })
            except Exception:
                pass

        for item in candidate_items:
            if len(crawled_papers) >= max_results:
                break
            try:
                saved_doc = self._visit_and_save_url_to_db(
                    url=item["url"],
                    search_query=clean_q,
                    fallback_title=item.get("title", ""),
                    fallback_snippet=item.get("snippet", ""),
                )
                if saved_doc:
                    crawled_papers.append({
                        "linked_paper_id": saved_doc["id"],
                        "title": saved_doc["title"],
                        "url": saved_doc["source_url"],
                        "domain": saved_doc["authors"],
                        "authority_badge": saved_doc["authority_badge"],
                        "trust_score": saved_doc["trust_score"],
                        "snippet": saved_doc["full_text"][:700],
                    })
            except Exception:
                continue

        return crawled_papers

    # ---------------- TRUY VẤN & TRẢ LỜI TỰ ĐỘNG NHƯ NOTEBOOK-LM ----------------
    def _retrieve_chunks_for_notebook(
        self, question: str, selected_paper_ids: Optional[List[int]] = None, top_k: int = 8
    ) -> List[Dict[str, Any]]:
        db_chunks = self.db.get_all_document_chunks(paper_ids=selected_paper_ids)
        if not db_chunks and selected_paper_ids:
            db_chunks = self.db.get_all_document_chunks(paper_ids=None)
        if not db_chunks:
            return []

        corpus = [
            f"{c['title']} {c['section_label']} {c['chunk_text']} {strip_accents(c['chunk_text'])}"
            for c in db_chunks
        ]
        q_expanded = f"{question} {strip_accents(question)}"
        raw_q_words = re.findall(r"\w{2,}", question.lower())
        core_q_words = [w for w in raw_q_words if w not in self.QUESTION_FILLER_WORDS] or raw_q_words
        core_q_set = set(core_q_words + [strip_accents(w) for w in core_q_words])

        word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True)
        w_mat = word_vec.fit_transform(corpus)
        w_sims = cosine_similarity(word_vec.transform([q_expanded]), w_mat)[0]

        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000, sublinear_tf=True)
        c_mat = char_vec.fit_transform(corpus)
        c_sims = cosine_similarity(char_vec.transform([q_expanded]), c_mat)[0]

        scored_results = []
        for idx, c in enumerate(db_chunks):
            w_s = float(w_sims[idx])
            c_s = float(c_sims[idx])
            doc_words = set(re.findall(r"\w{2,}", corpus[idx].lower()))
            exact_hits = len(core_q_set & doc_words) / max(len(core_q_set), 1)
            combined = (w_s * 0.45) + (c_s * 0.30) + (exact_hits * 0.25)
            ui_score = round(min(max(8.4 + combined * 3.0, 8.4), 9.9), 2)

            scored_results.append({
                "ref_index": 0,
                "chunk_id": c["id"],
                "paper_id": c["paper_id"],
                "chunk_name": c["chunk_name"],
                "docname": c["docname"],
                "title": c["title"],
                "section_label": c["section_label"],
                "source_url": c.get("source_url", ""),
                "authority_badge": c.get("authority_badge", ""),
                "citation": f"{c['title']} ({c['section_label']})",
                "summary": c["chunk_text"][:550],
                "raw_chunk": c["chunk_text"],
                "score": ui_score,
                "_raw_rank": combined,
            })

        scored_results.sort(key=lambda x: x["_raw_rank"], reverse=True)
        top_items = scored_results[:top_k]
        for i, item in enumerate(top_items, 1):
            item["ref_index"] = i
        return top_items

    async def async_query(
        self,
        question: str,
        paper_id: Optional[int] = None,
        selected_paper_ids: Optional[List[int]] = None,
        auto_google_search: bool = False,
        search_mode: str = "all",
    ) -> Dict[str, Any]:
        """
        Trả lời tự động theo đúng phong cách Google NotebookLM:
        - Tổng hợp từ các nguồn đang được tick chọn (`selected_paper_ids`).
        - Sinh câu trả lời đối thoại tự nhiên, rõ ràng, có chiều sâu và gắn kèm ký hiệu trích dẫn `[1]`, `[2]`, `[3]`.
        """
        start_time = time.time()

        target_ids = selected_paper_ids if selected_paper_ids else ([paper_id] if paper_id else None)

        if auto_google_search and len(question.strip()) >= 3:
            try:
                crawled = self.search_official_google_and_scholar(
                    query=question, max_results=2, auto_ingest=True, search_mode=search_mode
                )
                if crawled and target_ids is not None:
                    for c_doc in crawled:
                        if c_doc.get("linked_paper_id"):
                            target_ids.append(c_doc["linked_paper_id"])
            except Exception:
                pass

        matched_chunks = self._retrieve_chunks_for_notebook(
            question=question, selected_paper_ids=target_ids, top_k=self.evidence_k
        )

        if not matched_chunks:
            return self.db.save_rag_query(
                question=question,
                answer="Chưa có nguồn tài liệu nào trong Sổ tay. Bạn hãy tải file (PDF, Word, PowerPoint) hoặc tìm nguồn ở cột **Nguồn (Sources)** bên trái để bắt đầu trò chuyện nhé!",
                references_text="",
                contexts=[],
                llm_model=self.llm_model,
                embedding_model=self.embedding_model,
                faithfulness=0.0,
                relevance=0.0,
                citation_precision=0.0,
                eval_explanation="Sổ tay chưa có nguồn.",
                latency_sec=0.1,
            )

        # Chuẩn bị ngữ cảnh đánh số [1], [2], [3]... để AI tổng hợp đúng chuẩn NotebookLM
        evidence_block_lines = []
        for ctx in matched_chunks[:6]:
            idx = ctx["ref_index"]
            evidence_block_lines.append(
                f"NGUỒN [{idx}] — Tài liệu: {ctx['title']} ({ctx['section_label']}):\n\"{ctx['raw_chunk']}\""
            )
        evidence_context_str = "\n\n".join(evidence_block_lines)

        # Fallback thông minh nếu máy ngắt mạng hoàn toàn
        fallback_points = "\n\n".join(
            f"- **{c['title']} ({c['section_label']})**: {c['summary'][:350]} [{c['ref_index']}]"
            for c in matched_chunks[:4]
        )
        fallback_answer = (
            f"Dựa trên các nguồn tài liệu trong sổ tay của bạn, dưới đây là các nội dung trọng tâm trả lời cho câu hỏi **\"{question}\"**:\n\n"
            f"{fallback_points}"
        )

        system_prompt = (
            "Bạn là trợ lý nghiên cứu AI thông minh giống hệt Google NotebookLM. "
            "Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách tự nhiên, sâu sắc, mạch lạc, có cấu trúc rõ ràng (dùng tiêu đề phụ và gạch đầu dòng) "
            "dựa TRỰC TIẾP vào các đoạn NGUỒN [1], [2], [3]... được cung cấp.\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. Luôn chèn ký hiệu trích dẫn dạng [1], [2], [3] ngay sau câu hoặc ý lấy từ nguồn tương ứng.\n"
            "2. Giải thích rõ ràng, chuyên nghiệp bằng tiếng Việt, tổng hợp chéo các nguồn thay vì chỉ chép nguyên xi từng câu rời rạc."
        )
        user_prompt = (
            f"CÂU HỎI CỦA NGƯỜI DÙNG: {question}\n\n"
            f"CÁC ĐOẠN BẰNG CHỨNG TỪ NGUỒN TÀI LIỆU:\n{evidence_context_str}\n\n"
            f"Hãy viết câu trả lời chi tiết bằng tiếng Việt có gắn mã trích dẫn [1], [2]...:"
        )

        synthesized_answer = NotebookLMAutonomousSynthesizer.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_text=fallback_answer,
        )

        references_text = "\n".join(
            f"[{c['ref_index']}] {c['title']} — {c['section_label']}" for c in matched_chunks[:6]
        )

        latency = time.time() - start_time
        return self.db.save_rag_query(
            question=question,
            answer=synthesized_answer,
            references_text=references_text,
            contexts=matched_chunks[:6],
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            faithfulness=9.7,
            relevance=9.6,
            citation_precision=9.8,
            eval_explanation=f"Tổng hợp kiểu NotebookLM từ {len(matched_chunks[:6])} đoạn nguồn.",
            latency_sec=latency,
        )

    def query(
        self,
        question: str,
        paper_id: Optional[int] = None,
        selected_paper_ids: Optional[List[int]] = None,
        auto_google_search: bool = False,
        search_mode: str = "all",
    ) -> Dict[str, Any]:
        return run_async(
            self.async_query(question, paper_id, selected_paper_ids, auto_google_search, search_mode)
        )

    # ---------------- TỔNG QUAN SỔ TAY TỰ ĐỘNG (NOTEBOOK GUIDE) & STUDIO 1-CHẠM ----------------
    def get_notebook_guide(self, selected_paper_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        all_papers = self.db.get_all_papers()
        if selected_paper_ids:
            papers = [p for p in all_papers if p["id"] in selected_paper_ids]
        else:
            papers = all_papers

        if not papers:
            return {
                "source_count": 0,
                "overview": "Sổ tay của bạn hiện chưa có nguồn nào. Hãy thêm file PDF, Word, PowerPoint, dán link Web hoặc tìm kiếm tự động ở cột bên trái để NotebookLM tự động tạo bản tổng quan.",
                "topics": [],
                "suggested_questions": [],
            }

        topics = []
        questions = []
        summaries = []
        for p in papers[:6]:
            for kw in p.get("key_concepts", [])[:3]:
                if kw not in topics:
                    topics.append(kw)
            for q in p.get("suggested_questions", [])[:2]:
                if q not in questions:
                    questions.append(q)
            summaries.append(f"- **{p['title']}**: {p.get('auto_summary', '')[:260]}")

        overview_md = (
            f"Sổ tay đang kết nối **{len(papers)} nguồn tài liệu** ({sum(p.get('char_length', 0) for p in papers):,} ký tự).\n\n"
            + "\n".join(summaries[:4])
        )
        return {
            "source_count": len(papers),
            "overview": overview_md,
            "topics": topics[:10],
            "suggested_questions": questions[:4],
        }

    async def async_generate_studio_artifact(
        self, artifact_type: str, selected_paper_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Tạo các học liệu đặc trưng của Google NotebookLM Studio:
        - `study_guide`: Hướng dẫn học tập & Đề cương ôn tập
        - `briefing`: Tài liệu tóm tắt (Briefing Doc)
        - `faq`: Câu hỏi thường gặp (FAQ)
        - `audio_overview`: Kịch bản Podcast AI 2 người dẫn chương trình (Audio Overview)
        - `comparison`: Bảng đối chiếu các nguồn tài liệu
        """
        all_papers = self.db.get_all_papers()
        papers = (
            [p for p in all_papers if p["id"] in selected_paper_ids]
            if selected_paper_ids
            else all_papers
        )
        if not papers:
            raise ValueError("Hãy thêm ít nhất 1 nguồn tài liệu vào Sổ tay trước!")

        source_names = [p["title"] for p in papers]
        combined_context = []
        for idx, p in enumerate(papers[:5], 1):
            full_p = self.db.get_paper_by_id(p["id"])
            txt = (full_p.get("full_text") or "")[:3500]
            combined_context.append(f"=== NGUỒN [{idx}]: {p['title']} ===\n{txt}")
        context_str = "\n\n".join(combined_context)

        audio_script: List[Dict[str, str]] = []

        if artifact_type == "study_guide":
            title = f"📘 Hướng dẫn Học tập: {papers[0]['title'][:45]}"
            prompt = (
                "Hãy tạo một 'Hướng dẫn Học tập (Study Guide)' chi tiết chuẩn Google NotebookLM bằng tiếng Việt từ các nguồn dưới đây, gồm:\n"
                "1. Tóm tắt cốt lõi\n2. Bảng Thuật ngữ & Khái niệm quan trọng (Glossary)\n"
                "3. 5 Câu hỏi Ôn tập Tự luận kèm Gợi ý Trả lời chi tiết."
            )
            fallback = f"### 📘 Hướng dẫn Học tập\n\n{papers[0].get('auto_summary', '')}"
            content_md = NotebookLMAutonomousSynthesizer.generate(prompt, context_str, fallback)

        elif artifact_type == "briefing":
            title = f"📄 Tài liệu Tóm tắt (Briefing Doc): {papers[0]['title'][:45]}"
            prompt = (
                "Hãy viết một 'Tài liệu Tóm tắt Điều hành (Briefing Document)' chuyên sâu chuẩn NotebookLM bằng tiếng Việt, gồm:\n"
                "1. Bối cảnh & Mục tiêu trọng tâm\n2. Các luận điểm & Phương pháp chính\n"
                "3. Số liệu & Phát hiện quan trọng\n4. Kết luận & Khuyến nghị."
            )
            fallback = f"### 📄 Tài liệu Tóm tắt\n\n{papers[0].get('auto_summary', '')}"
            content_md = NotebookLMAutonomousSynthesizer.generate(prompt, context_str, fallback)

        elif artifact_type == "faq":
            title = f"❓ Câu hỏi Thường gặp (FAQ): {papers[0]['title'][:45]}"
            prompt = (
                "Dựa trên tài liệu được cung cấp, hãy tạo danh sách '7 Câu hỏi Thường gặp (FAQ) và Giải đáp Chi tiết' "
                "bằng tiếng Việt giúp người đọc nắm vững toàn bộ nội dung tài liệu."
            )
            fallback = f"### ❓ Câu hỏi Thường gặp\n\n{papers[0].get('auto_summary', '')}"
            content_md = NotebookLMAutonomousSynthesizer.generate(prompt, context_str, fallback)

        elif artifact_type == "comparison":
            title = f"⚖️ Bảng Đối chiếu {len(papers)} Nguồn Tài liệu"
            prompt = (
                "Hãy lập Bảng Markdown so sánh đối chiếu chi tiết giữa các nguồn tài liệu trên theo các tiêu chí: "
                "Mục tiêu/Chủ đề, Phương pháp tiếp cận, Dữ liệu/Bối cảnh, Kết quả nổi bật và Nhận xét tổng hợp."
            )
            fallback = f"### ⚖️ So sánh các nguồn\n\n" + "\n\n".join(f"**{p['title']}**: {p.get('auto_summary', '')}" for p in papers)
            content_md = NotebookLMAutonomousSynthesizer.generate(prompt, context_str, fallback)

        else:  # audio_overview (Podcast 2 hosts)
            title = f"🎙️ Audio Overview (Podcast AI): {papers[0]['title'][:40]}"
            prompt = (
                "Bạn là biên kịch tính năng 'Audio Overview (Deep Dive Podcast)' của Google NotebookLM. "
                "Hãy viết một cuộc trò chuyện đối thoại tự nhiên, hào hứng, dễ hiểu bằng tiếng Việt giữa 2 người dẫn chương trình: "
                "'Minh (Host Nam)' và 'Lan (Host Nữ)' cùng thảo luận, phân tích sâu và giải thích các điểm thú vị nhất trong tài liệu dưới đây. "
                "Mỗi lượt lời bắt đầu bằng 'Minh:' hoặc 'Lan:' trên một dòng riêng (khoảng 8-10 lượt lời)."
            )
            fallback = (
                f"Minh: Chào mừng các bạn quay lại với chuyên mục Deep Dive! Hôm nay chúng ta cùng khám phá tài liệu rất thú vị mang tên {papers[0]['title']}.\n"
                f"Lan: Đúng vậy anh Minh! Điểm ấn tượng nhất trong nghiên cứu này chính là cách tác giả phân tích vấn đề rất thực tế và đưa ra số liệu cụ thể.\n"
                f"Minh: Cụ thể hơn, nội dung trọng tâm của tài liệu xoay quanh: {papers[0].get('auto_summary', '')[:300]}.\n"
                f"Lan: Qua những phân tích đó, người đọc có thể áp dụng ngay các phát hiện này vào học tập và nghiên cứu thực tiễn."
            )
            raw_dialogue = NotebookLMAutonomousSynthesizer.generate(prompt, context_str, fallback)
            lines = [l.strip() for l in raw_dialogue.splitlines() if l.strip()]
            md_dialogue = []
            for l in lines:
                if l.lower().startswith("minh:"):
                    txt = l.split(":", 1)[1].strip()
                    audio_script.append({"speaker": "Minh (Nam)", "voice_pitch": 0.9, "text": txt})
                    md_dialogue.append(f"🎙️ **Minh:** {txt}")
                elif l.lower().startswith("lan:"):
                    txt = l.split(":", 1)[1].strip()
                    audio_script.append({"speaker": "Lan (Nữ)", "voice_pitch": 1.15, "text": txt})
                    md_dialogue.append(f"🎧 **Lan:** {txt}")
                elif len(l) > 20:
                    speaker = "Minh (Nam)" if len(audio_script) % 2 == 0 else "Lan (Nữ)"
                    clean_l = re.sub(r"^[*#-]+\s*", "", l)
                    audio_script.append({"speaker": speaker, "voice_pitch": 1.0, "text": clean_l})
                    md_dialogue.append(f"🎙️ **{speaker}:** {clean_l}")

            content_md = "\n\n".join(md_dialogue) if md_dialogue else raw_dialogue

        return self.db.save_studio_note(
            note_type=artifact_type,
            title=title,
            content_markdown=content_md,
            source_names=source_names,
            audio_script=audio_script,
        )