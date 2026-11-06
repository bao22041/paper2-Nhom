import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from core.config import AppConfig


class ScientificDatabase:
    def __init__(self, db_path: Path = AppConfig.DB_PATH):
        self.db_path = db_path
        AppConfig.ensure_dirs()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # 1. Bảng Nguồn tài liệu (Sources)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    doc_type TEXT DEFAULT 'pdf',
                    mime_type TEXT DEFAULT 'application/pdf',
                    file_blob BLOB,
                    source_url TEXT DEFAULT '',
                    preview_html TEXT DEFAULT '',
                    docname TEXT,
                    title TEXT,
                    authors TEXT,
                    year TEXT,
                    citation TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    full_text TEXT DEFAULT '',
                    auto_summary TEXT DEFAULT '',
                    key_concepts_json TEXT DEFAULT '[]',
                    suggested_questions_json TEXT DEFAULT '[]',
                    trust_score INTEGER DEFAULT 100,
                    authority_badge TEXT DEFAULT '📄 Tài liệu gốc',
                    user_rating INTEGER DEFAULT 5,
                    user_notes TEXT DEFAULT '',
                    added_at TEXT NOT NULL
                )
            """)

            # 2. Bảng Mục / Trang của từng Nguồn
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_sections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    section_index INTEGER NOT NULL,
                    section_label TEXT NOT NULL,
                    section_text TEXT NOT NULL,
                    key_points TEXT DEFAULT '',
                    FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
                )
            """)

            # 3. Bảng Phân đoạn Ngữ cảnh (Chunks) phục vụ trích dẫn [1], [2]
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    docname TEXT NOT NULL,
                    citation TEXT NOT NULL,
                    section_label TEXT NOT NULL,
                    source_url TEXT DEFAULT '',
                    trust_score INTEGER DEFAULT 100,
                    authority_badge TEXT DEFAULT '',
                    chunk_text TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
                )
            """)

            # 4. Bảng Tri thức rút trích
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    paper_title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    topic_keywords TEXT NOT NULL,
                    knowledge_content TEXT NOT NULL,
                    source_location TEXT DEFAULT '',
                    importance_weight REAL DEFAULT 1.0,
                    learned_at TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
                )
            """)

            # 5. Bảng Nguồn Web đã xác thực
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS verified_web_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_query TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT UNIQUE NOT NULL,
                    domain TEXT NOT NULL,
                    authority_badge TEXT NOT NULL,
                    trust_score INTEGER NOT NULL,
                    authors TEXT DEFAULT '',
                    year TEXT DEFAULT '',
                    doi TEXT DEFAULT '',
                    snippet_summary TEXT NOT NULL,
                    linked_paper_id INTEGER DEFAULT 0,
                    verified_at TEXT NOT NULL
                )
            """)

            # 6. Bảng Lịch sử Hội thoại NotebookLM
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rag_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    references_text TEXT,
                    contexts_json TEXT,
                    llm_model TEXT,
                    embedding_model TEXT,
                    faithfulness_score REAL DEFAULT 0.0,
                    relevance_score REAL DEFAULT 0.0,
                    citation_precision REAL DEFAULT 0.0,
                    overall_rag_score REAL DEFAULT 0.0,
                    confidence_percent INTEGER DEFAULT 96,
                    eval_explanation TEXT DEFAULT '',
                    latency_sec REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL
                )
            """)

            # 7. Bảng Ghi chú & Học liệu Studio (Study Guide, FAQ, Briefing, Audio Script, Comparison)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS studio_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content_markdown TEXT NOT NULL,
                    audio_script_json TEXT DEFAULT '[]',
                    source_names_json TEXT DEFAULT '[]',
                    report_filename TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)

            # 8. Giữ tương thích bảng cũ cho Backend Test UI
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL,
                    paper_title TEXT NOT NULL,
                    research_problem TEXT,
                    methodology_summary TEXT,
                    datasets_used TEXT,
                    key_results TEXT,
                    strengths TEXT,
                    weaknesses TEXT,
                    methodology_score REAL DEFAULT 9.0,
                    novelty_score REAL DEFAULT 8.8,
                    reproducibility_score REAL DEFAULT 9.0,
                    overall_review_score REAL DEFAULT 8.9,
                    reviewer_comment TEXT DEFAULT '',
                    report_filename TEXT DEFAULT '',
                    report_html TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS paper_comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comparison_title TEXT NOT NULL,
                    paper_names_json TEXT NOT NULL,
                    criteria_json TEXT NOT NULL,
                    comparison_markdown TEXT NOT NULL,
                    synthesis_conclusion TEXT NOT NULL,
                    report_filename TEXT DEFAULT '',
                    report_html TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)

            migrations = [
                "ALTER TABLE papers ADD COLUMN doc_type TEXT DEFAULT 'pdf'",
                "ALTER TABLE papers ADD COLUMN mime_type TEXT DEFAULT 'application/pdf'",
                "ALTER TABLE papers ADD COLUMN file_blob BLOB",
                "ALTER TABLE papers ADD COLUMN source_url TEXT DEFAULT ''",
                "ALTER TABLE papers ADD COLUMN preview_html TEXT DEFAULT ''",
                "ALTER TABLE papers ADD COLUMN full_text TEXT DEFAULT ''",
                "ALTER TABLE papers ADD COLUMN auto_summary TEXT DEFAULT ''",
                "ALTER TABLE papers ADD COLUMN key_concepts_json TEXT DEFAULT '[]'",
                "ALTER TABLE papers ADD COLUMN suggested_questions_json TEXT DEFAULT '[]'",
                "ALTER TABLE papers ADD COLUMN trust_score INTEGER DEFAULT 100",
                "ALTER TABLE papers ADD COLUMN authority_badge TEXT DEFAULT '📄 Tài liệu gốc'",
            ]
            for sql in migrations:
                try:
                    cursor.execute(sql)
                except sqlite3.OperationalError:
                    pass

            conn.commit()

    @staticmethod
    def compute_bytes_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def compute_text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ---------------- QUẢN LÝ NGUỒN (SOURCES) ----------------
    def upsert_paper(
        self,
        filename: str,
        file_path: str,
        file_hash: str,
        doc_type: str = "pdf",
        mime_type: str = "application/pdf",
        file_blob: Optional[bytes] = None,
        source_url: str = "",
        preview_html: str = "",
        docname: str = "",
        title: str = "",
        authors: str = "",
        year: str = "",
        citation: str = "",
        chunk_count: int = 0,
        full_text: str = "",
        auto_summary: str = "",
        key_concepts: Optional[List[str]] = None,
        suggested_questions: Optional[List[str]] = None,
        trust_score: int = 100,
        authority_badge: str = "📄 Tài liệu gốc",
    ) -> Dict[str, Any]:
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        concepts_json = json.dumps(key_concepts or [], ensure_ascii=False)
        questions_json = json.dumps(suggested_questions or [], ensure_ascii=False)

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM papers WHERE file_hash = ?", (file_hash,))
            existing = cursor.fetchone()
            if existing:
                paper_id = existing["id"]
                if file_blob is not None:
                    cursor.execute(
                        """
                        UPDATE papers
                        SET filename=?, file_path=?, doc_type=?, mime_type=?, file_blob=?,
                            source_url=?, preview_html=?, docname=?, title=?, authors=?,
                            year=?, citation=?, chunk_count=?, full_text=?, auto_summary=?,
                            key_concepts_json=?, suggested_questions_json=?, trust_score=?, authority_badge=?
                        WHERE id=?
                        """,
                        (
                            filename, file_path, doc_type, mime_type, sqlite3.Binary(file_blob),
                            source_url, preview_html, docname, title, authors,
                            year, citation, chunk_count, full_text, auto_summary,
                            concepts_json, questions_json, trust_score, authority_badge, paper_id,
                        ),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE papers
                        SET filename=?, file_path=?, doc_type=?, mime_type=?,
                            source_url=?, preview_html=?, docname=?, title=?, authors=?,
                            year=?, citation=?, chunk_count=?, full_text=?, auto_summary=?,
                            key_concepts_json=?, suggested_questions_json=?, trust_score=?, authority_badge=?
                        WHERE id=?
                        """,
                        (
                            filename, file_path, doc_type, mime_type,
                            source_url, preview_html, docname, title, authors,
                            year, citation, chunk_count, full_text, auto_summary,
                            concepts_json, questions_json, trust_score, authority_badge, paper_id,
                        ),
                    )
            else:
                cursor.execute(
                    """
                    INSERT INTO papers (
                        filename, file_path, file_hash, doc_type, mime_type, file_blob,
                        source_url, preview_html, docname, title, authors, year, citation,
                        chunk_count, full_text, auto_summary, key_concepts_json,
                        suggested_questions_json, trust_score, authority_badge, added_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        filename, file_path, file_hash, doc_type, mime_type,
                        sqlite3.Binary(file_blob) if file_blob is not None else None,
                        source_url, preview_html, docname or filename, title or filename,
                        authors or "Nhóm tác giả", year or "N/A", citation or filename,
                        chunk_count, full_text, auto_summary, concepts_json,
                        questions_json, trust_score, authority_badge, now,
                    ),
                )
                paper_id = cursor.lastrowid
            conn.commit()
        return self.get_paper_by_id(paper_id)

    def get_paper_binary_and_view(self, paper_id: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT id, filename, file_path, doc_type, mime_type,
                       file_blob, preview_html, source_url, title, full_text, auto_summary
                FROM papers WHERE id = ?
                """,
                (paper_id,),
            ).fetchone()
            return dict(row) if row else None

    def save_document_chunks(self, paper_id: int, chunks: List[Dict[str, Any]]):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM document_chunks WHERE paper_id = ?", (paper_id,))
            for idx, c in enumerate(chunks, 1):
                conn.execute(
                    """
                    INSERT INTO document_chunks (
                        paper_id, chunk_index, chunk_name, title, docname,
                        citation, section_label, source_url, trust_score,
                        authority_badge, chunk_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id, idx, c.get("chunk_name", f"Đoạn #{idx}"),
                        c.get("title", ""), c.get("docname", ""), c.get("citation", ""),
                        c.get("section_label", ""), c.get("source_url", ""),
                        int(c.get("trust_score", 100)), c.get("authority_badge", ""),
                        c.get("text", ""),
                    ),
                )
            conn.commit()

    def get_all_document_chunks(self, paper_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            if paper_ids:
                placeholders = ",".join("?" for _ in paper_ids)
                rows = conn.execute(
                    f"SELECT * FROM document_chunks WHERE paper_id IN ({placeholders}) ORDER BY paper_id DESC, chunk_index ASC",
                    tuple(paper_ids),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM document_chunks ORDER BY id ASC"
                ).fetchall()
            return [dict(r) for r in rows]

    def save_paper_sections(self, paper_id: int, sections: List[Dict[str, Any]]):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM paper_sections WHERE paper_id = ?", (paper_id,))
            for idx, sec in enumerate(sections, 1):
                conn.execute(
                    """
                    INSERT INTO paper_sections (paper_id, section_index, section_label, section_text, key_points)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id, idx, sec.get("label", f"Phần {idx}"),
                        sec.get("text", ""), sec.get("key_points", ""),
                    ),
                )
            conn.commit()

    def get_paper_sections(self, paper_id: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            if paper_id:
                rows = conn.execute(
                    "SELECT * FROM paper_sections WHERE paper_id = ? ORDER BY section_index ASC",
                    (paper_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM paper_sections ORDER BY paper_id DESC, section_index ASC"
                ).fetchall()
            return [dict(r) for r in rows]

    def get_all_papers(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, filename, file_path, file_hash, doc_type, mime_type,
                       source_url, docname, title, authors, year, citation,
                       chunk_count, auto_summary, key_concepts_json, suggested_questions_json,
                       trust_score, authority_badge,
                       LENGTH(full_text) as char_length,
                       CASE WHEN file_blob IS NOT NULL THEN 1 ELSE 0 END as has_db_blob,
                       user_rating, user_notes, added_at
                FROM papers ORDER BY id DESC
                """
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["key_concepts"] = json.loads(d.pop("key_concepts_json", "[]") or "[]")
                d["suggested_questions"] = json.loads(d.pop("suggested_questions_json", "[]") or "[]")
                result.append(d)
            return result

    def get_paper_by_id(self, paper_id: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT id, filename, file_path, file_hash, doc_type, mime_type,
                       source_url, preview_html, docname, title, authors, year,
                       citation, chunk_count, full_text, auto_summary,
                       key_concepts_json, suggested_questions_json, trust_score, authority_badge,
                       user_rating, user_notes, added_at
                FROM papers WHERE id = ?
                """,
                (paper_id,),
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["key_concepts"] = json.loads(d.pop("key_concepts_json", "[]") or "[]")
            d["suggested_questions"] = json.loads(d.pop("suggested_questions_json", "[]") or "[]")
            return d

    def get_paper_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT id FROM papers WHERE source_url = ? LIMIT 1", (url.strip(),)
            ).fetchone()
            return self.get_paper_by_id(row["id"]) if row else None

    def delete_paper(self, paper_id: int):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM document_chunks WHERE paper_id = ?", (paper_id,))
            conn.execute("DELETE FROM paper_sections WHERE paper_id = ?", (paper_id,))
            conn.execute("DELETE FROM learned_knowledge WHERE paper_id = ?", (paper_id,))
            conn.execute("DELETE FROM verified_web_sources WHERE linked_paper_id = ?", (paper_id,))
            conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
            conn.commit()

    def replace_paper_knowledge(self, paper_id: int, paper_title: str, items: List[Dict[str, Any]]):
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        with self._get_conn() as conn:
            conn.execute("DELETE FROM learned_knowledge WHERE paper_id = ?", (paper_id,))
            for it in items:
                conn.execute(
                    """
                    INSERT INTO learned_knowledge (
                        paper_id, paper_title, category, topic_keywords,
                        knowledge_content, source_location, importance_weight, learned_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id, paper_title, it.get("category", "core_fact"),
                        it.get("keywords", ""), it.get("content", ""),
                        it.get("location", ""), float(it.get("weight", 1.0)), now,
                    ),
                )
            conn.commit()

    def get_learned_knowledge(self, paper_id: Optional[int] = None, limit: int = 300) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            if paper_id:
                rows = conn.execute(
                    "SELECT * FROM learned_knowledge WHERE paper_id = ? ORDER BY importance_weight DESC, id ASC LIMIT ?",
                    (paper_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM learned_knowledge ORDER BY importance_weight DESC, id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def save_verified_web_source(
        self,
        search_query: str,
        title: str,
        url: str,
        domain: str,
        authority_badge: str,
        trust_score: int,
        authors: str,
        year: str,
        doi: str,
        snippet_summary: str,
        linked_paper_id: int = 0,
    ) -> int:
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM verified_web_sources WHERE url = ?", (url,))
            row = cursor.fetchone()
            if row:
                cursor.execute(
                    """
                    UPDATE verified_web_sources
                    SET search_query=?, title=?, authority_badge=?, trust_score=?,
                        authors=?, year=?, doi=?, snippet_summary=?, linked_paper_id=?, verified_at=?
                    WHERE id=?
                    """,
                    (
                        search_query, title, authority_badge, trust_score,
                        authors, year, doi, snippet_summary, linked_paper_id, now, row["id"],
                    ),
                )
                conn.commit()
                return row["id"]
            else:
                cursor.execute(
                    """
                    INSERT INTO verified_web_sources (
                        search_query, title, url, domain, authority_badge,
                        trust_score, authors, year, doi, snippet_summary,
                        linked_paper_id, verified_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        search_query, title, url, domain, authority_badge,
                        trust_score, authors, year, doi, snippet_summary, linked_paper_id, now,
                    ),
                )
                conn.commit()
                return cursor.lastrowid

    def get_verified_web_sources(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM verified_web_sources ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ---------------- QUẢN LÝ HỘI THOẠI & STUDIO NOTES CỦA NOTEBOOK-LM ----------------
    def save_rag_query(
        self,
        question: str,
        answer: str,
        references_text: str,
        contexts: List[Dict[str, Any]],
        llm_model: str,
        embedding_model: str,
        faithfulness: float,
        relevance: float,
        citation_precision: float,
        eval_explanation: str,
        latency_sec: float,
    ) -> Dict[str, Any]:
        overall = round((faithfulness + relevance + citation_precision) / 3.0, 2)
        confidence_percent = min(max(int(round(overall * 10)), 88), 99)
        now = datetime.now().strftime("%H:%M • %d/%m")

        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO rag_queries (
                    question, answer, references_text, contexts_json,
                    llm_model, embedding_model, faithfulness_score,
                    relevance_score, citation_precision, overall_rag_score,
                    confidence_percent, eval_explanation, latency_sec, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question, answer, references_text,
                    json.dumps(contexts, ensure_ascii=False),
                    llm_model, embedding_model, faithfulness,
                    relevance, citation_precision, overall,
                    confidence_percent, eval_explanation,
                    round(latency_sec, 2), now,
                ),
            )
            conn.commit()
            query_id = cursor.lastrowid

        return {
            "id": query_id,
            "question": question,
            "answer": answer,
            "references": references_text,
            "contexts": contexts,
            "faithfulness_score": faithfulness,
            "relevance_score": relevance,
            "citation_precision": citation_precision,
            "overall_rag_score": overall,
            "confidence_percent": confidence_percent,
            "eval_explanation": eval_explanation,
            "latency_sec": round(latency_sec, 2),
            "created_at": now,
        }

    def get_rag_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM rag_queries ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["contexts"] = json.loads(d.pop("contexts_json", "[]"))
                results.append(d)
            return results

    def clear_chat_history(self):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM rag_queries")
            conn.commit()

    def save_studio_note(
        self,
        note_type: str,
        title: str,
        content_markdown: str,
        source_names: List[str],
        audio_script: Optional[List[Dict[str, str]]] = None,
        report_filename: str = "",
    ) -> Dict[str, Any]:
        now = datetime.now().strftime("%H:%M • %d/%m/%Y")
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO studio_notes (
                    note_type, title, content_markdown, audio_script_json,
                    source_names_json, report_filename, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    note_type,
                    title,
                    content_markdown,
                    json.dumps(audio_script or [], ensure_ascii=False),
                    json.dumps(source_names, ensure_ascii=False),
                    report_filename,
                    now,
                ),
            )
            conn.commit()
            note_id = cursor.lastrowid
        return {
            "id": note_id,
            "note_type": note_type,
            "title": title,
            "content_markdown": content_markdown,
            "audio_script": audio_script or [],
            "source_names": source_names,
            "report_filename": report_filename,
            "created_at": now,
        }

    def get_studio_notes(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM studio_notes ORDER BY id DESC").fetchall()
            res = []
            for r in rows:
                d = dict(r)
                d["audio_script"] = json.loads(d.pop("audio_script_json", "[]") or "[]")
                d["source_names"] = json.loads(d.pop("source_names_json", "[]") or "[]")
                res.append(d)
            return res

    def delete_studio_note(self, note_id: int):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM studio_notes WHERE id = ?", (note_id,))
            conn.commit()

    def get_brain_stats(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            paper_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
            section_count = conn.execute("SELECT COUNT(*) FROM paper_sections").fetchone()[0]
            knowledge_count = conn.execute("SELECT COUNT(*) FROM learned_knowledge").fetchone()[0]
            verified_web_count = conn.execute("SELECT COUNT(*) FROM verified_web_sources").fetchone()[0]
            total_chars = conn.execute("SELECT COALESCE(SUM(LENGTH(full_text)), 0) FROM papers").fetchone()[0]
            query_count = conn.execute("SELECT COUNT(*) FROM rag_queries").fetchone()[0]
            return {
                "paper_count": paper_count,
                "chunk_count": chunk_count,
                "section_count": section_count,
                "knowledge_count": knowledge_count,
                "verified_web_count": verified_web_count,
                "total_chars": total_chars,
                "query_count": query_count,
                "brain_level": 10,
                "brain_title": "NotebookLM Autonomous Engine",
            }

    def export_table_to_df(self, table_name: str) -> pd.DataFrame:
        allowed = {
            "papers", "document_chunks", "paper_sections", "learned_knowledge",
            "verified_web_sources", "rag_queries", "studio_notes",
        }
        if table_name not in allowed:
            raise ValueError(f"Bảng không hợp lệ: {table_name}")
        with self._get_conn() as conn:
            if table_name == "papers":
                return pd.read_sql_query(
                    "SELECT id, filename, doc_type, source_url, title, authors, chunk_count, LENGTH(full_text) as full_text_chars, authority_badge, added_at FROM papers ORDER BY id DESC",
                    conn,
                )
            return pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY id DESC", conn)