import webbrowser
from pathlib import Path
from typing import List, Optional
from threading import Timer

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response, FileResponse, RedirectResponse
from pydantic import BaseModel

from core.config import AppConfig
from core.database import ScientificDatabase
from core.rag_engine import PaperQA2UnifiedEngine

app = FastAPI(title="THIÊN BẢO NOTEBOOK-LM v7.0")
db = ScientificDatabase()
engine = PaperQA2UnifiedEngine(db=db)


class WebUrlPayload(BaseModel):
    url: str


class PastedTextPayload(BaseModel):
    title: str
    content: str


class OfficialSearchPayload(BaseModel):
    query: str
    max_results: int = 3
    search_mode: str = "all"


class QuestionPayload(BaseModel):
    question: str
    paper_id: Optional[int] = None
    selected_paper_ids: Optional[List[int]] = None
    auto_google_search: bool = False
    search_mode: str = "all"


class StudioGeneratePayload(BaseModel):
    artifact_type: str
    selected_paper_ids: Optional[List[int]] = None


class CustomNotePayload(BaseModel):
    title: str
    content_markdown: str


@app.get("/", response_class=HTMLResponse)
async def serve_user_ui():
    html_file = AppConfig.BASE_DIR / "frontend" / "index.html"
    return html_file.read_text(encoding="utf-8")


@app.post("/api/notebook/guide")
async def get_notebook_guide(payload: StudioGeneratePayload):
    return engine.get_notebook_guide(payload.selected_paper_ids)


@app.get("/api/papers")
async def list_papers():
    return db.get_all_papers()


@app.get("/api/papers/{paper_id}/detail")
async def get_paper_detail(paper_id: int):
    paper = db.get_paper_by_id(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Không tìm thấy nguồn tài liệu.")
    sections = db.get_paper_sections(paper_id)
    chunks = db.get_all_document_chunks(paper_ids=[paper_id])
    return {
        "paper": paper,
        "sections": sections,
        "chunks": chunks,
    }


@app.post("/api/papers/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        save_path = AppConfig.PDF_DIR / file.filename
        with open(save_path, "wb") as f:
            f.write(await file.read())
        return await engine.async_add_document(save_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi khi tải tệp: {e}")


@app.post("/api/papers/add-url")
async def add_web_url(payload: WebUrlPayload):
    try:
        return await engine.async_add_web_url(payload.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/papers/add-text")
async def add_pasted_text(payload: PastedTextPayload):
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="Nội dung văn bản không được để trống.")
    return await engine.async_add_pasted_text(payload.title, payload.content)


@app.post("/api/search/official")
async def search_official_sources(payload: OfficialSearchPayload):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập chủ đề cần khám phá.")
    found = engine.search_official_google_and_scholar(
        query=payload.query.strip(),
        max_results=payload.max_results,
        auto_ingest=True,
        search_mode=payload.search_mode,
    )
    return {"query": payload.query, "count": len(found), "sources": found}


@app.delete("/api/papers/{paper_id}")
async def remove_paper(paper_id: int):
    db.delete_paper(paper_id)
    engine.remove_paper_from_brain(paper_id)
    return {"status": "deleted"}


@app.get("/api/files/view/{paper_id}")
async def view_document_from_database(paper_id: int):
    rec = db.get_paper_binary_and_view(paper_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Không tìm thấy nguồn.")
    doc_type = (rec.get("doc_type") or "pdf").lower()
    if doc_type in {"docx", "pptx", "web", "txt"} and rec.get("preview_html"):
        return HTMLResponse(content=rec["preview_html"])
    if rec.get("file_blob"):
        return Response(
            content=bytes(rec["file_blob"]),
            media_type=rec.get("mime_type") or "application/pdf",
            headers={"Content-Disposition": f'inline; filename="{rec["filename"]}"'},
        )
    file_path = Path(rec["file_path"])
    if file_path.exists():
        return FileResponse(path=str(file_path), media_type="application/pdf")
    raise HTTPException(status_code=404, detail="Không tìm thấy tệp.")


@app.post("/api/chat/ask")
async def ask_question(payload: QuestionPayload):
    try:
        return await engine.async_query(
            question=payload.question,
            paper_id=payload.paper_id,
            selected_paper_ids=payload.selected_paper_ids,
            auto_google_search=payload.auto_google_search,
            search_mode=payload.search_mode,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/chat/history")
async def get_chat_history():
    return db.get_rag_queries(limit=40)


@app.delete("/api/chat/clear")
async def clear_chat():
    db.clear_chat_history()
    return {"status": "cleared"}


@app.post("/api/studio/generate")
async def generate_studio_artifact(payload: StudioGeneratePayload):
    try:
        return await engine.async_generate_studio_artifact(
            artifact_type=payload.artifact_type,
            selected_paper_ids=payload.selected_paper_ids,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/studio/pin-note")
async def pin_custom_note(payload: CustomNotePayload):
    return db.save_studio_note(
        note_type="pinned_chat",
        title=payload.title,
        content_markdown=payload.content_markdown,
        source_names=["Trò chuyện NotebookLM"],
    )


@app.get("/api/studio/notes")
async def list_studio_notes():
    return db.get_studio_notes()


@app.delete("/api/studio/notes/{note_id}")
async def remove_studio_note(note_id: int):
    db.delete_studio_note(note_id)
    return {"status": "deleted"}


if __name__ == "__main__":
    Timer(1.5, lambda: webbrowser.open("http://localhost:8000")).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)