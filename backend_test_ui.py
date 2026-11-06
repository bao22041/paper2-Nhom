import streamlit as st
import pandas as pd

from core.config import AppConfig
from core.database import ScientificDatabase
from core.rag_engine import PaperQA2UnifiedEngine, OfficialSourceValidator

st.set_page_config(
    page_title="THIÊN BẢO AI v6.0 - Database & Deep Crawler Lab",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

db = ScientificDatabase()
engine = PaperQA2UnifiedEngine(db=db)
stats = db.get_brain_stats()

st.markdown(
    f"""
    <div style="background:linear-gradient(135deg,#0f172a,#1e3a8a);color:white;padding:1.2rem 1.6rem;border-radius:12px;margin-bottom:1rem;">
        <h2 style="margin:0;color:white;">🗄️ THIÊN BẢO AI v6.0 — Hệ thống Quản trị Database Tập trung & Cào Toàn Văn Link Google (Port 8501)</h2>
        <p style="margin:0.3rem 0 0 0;color:#cbd5e1;font-size:0.9rem;">
            Đã lưu trong SQLite: <b>{stats['paper_count']}</b> File & Trang Web •
            <b>{stats['chunk_count']}</b> đoạn văn bản (`document_chunks`) •
            <b>{stats['total_chars']:,}</b> ký tự toàn văn •
            <b>{stats['verified_web_count']}</b> Link Google đã vào đọc toàn văn
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.link_button(
        "🌐 Mở Giao diện Người dùng (Port 8000)",
        "http://localhost:8000",
        use_container_width=True,
    )
    if st.button("🔄 Đồng bộ toàn bộ File vào SQLite Database", type="primary", use_container_width=True):
        res = engine.sync_and_learn_all_files()
        st.success(f"Đã lưu {res['total_chunks']} đoạn văn bản vào bảng document_chunks!")
        st.rerun()

    st.divider()
    st.subheader("🌐 Test Tìm Google & Vào thẳng Link đọc toàn văn")
    test_search_q = st.text_input("Nhập chủ đề hoặc câu hỏi:")
    if st.button("🔍 Tìm & Vào Link Đọc Lưu vào DB", use_container_width=True):
        if test_search_q.strip():
            with st.spinner("Đang tìm trên Google, vào từng link đọc toàn bộ bài viết và lưu vào SQLite..."):
                found = engine.search_official_google_and_scholar(test_search_q.strip(), max_results=4)
                st.success(f"Đã vào đọc và lưu {len(found)} trang web vào bảng papers & document_chunks!")
                st.rerun()

tab_test_rag, tab_db_chunks, tab_verified_web, tab_sqlite = st.tabs(
    [
        "🔬 1. Test Hỏi đáp Trực tiếp từ Database",
        "🧩 2. Các đoạn văn bản lưu trong SQLite (`document_chunks`)",
        "🌐 3. Các Link Google Đã Vào Đọc Toàn Văn",
        "🗄️ 4. Xuất Dữ liệu các Bảng SQLite",
    ]
)

with tab_test_rag:
    st.subheader("🔬 Kiểm thử Hỏi đáp (Tự động vào Link Google đọc nội dung + Truy vấn SQLite)")
    auto_gg = st.checkbox("🌐 Tự động tìm trên Google, vào link đọc nội dung và lưu vào Database trước khi trả lời", value=True)
    q_test = st.text_area("Nhập câu hỏi:", value="Cảm xúc học tập trực tuyến của sinh viên bị ảnh hưởng bởi những yếu tố nào?")
    if st.button("▶️ Chạy Truy vấn", type="primary"):
        with st.spinner("Đang xử lý..."):
            res = engine.query(q_test, auto_google_search=auto_gg)
            st.session_state["test_rag_out"] = res

    if "test_rag_out" in st.session_state:
        r = st.session_state["test_rag_out"]
        st.info(r["eval_explanation"])
        st.markdown(r["answer"])

with tab_db_chunks:
    st.subheader("🧩 Toàn bộ các đoạn văn bản (`document_chunks`) đang lưu trong SQLite Database")
    chunks = db.get_all_document_chunks()
    if chunks:
        st.dataframe(pd.DataFrame(chunks), use_container_width=True)
    else:
        st.info("Chưa có đoạn văn bản nào trong Database.")

with tab_verified_web:
    st.subheader("🌐 Danh sách các Link Google đã truy cập vào đọc toàn văn")
    v_rows = db.get_verified_web_sources(limit=100)
    if v_rows:
        st.dataframe(pd.DataFrame(v_rows), use_container_width=True)

with tab_sqlite:
    st.subheader("🗄️ Xem & Xuất các Bảng trong SQLite (`scientific_rag.db`)")
    tb = st.selectbox(
        "Chọn bảng:",
        [
            "papers",
            "document_chunks",
            "paper_sections",
            "learned_knowledge",
            "verified_web_sources",
            "rag_queries",
            "paper_reviews",
            "paper_comparisons",
        ],
    )
    df_tb = db.export_table_to_df(tb)
    st.dataframe(df_tb, use_container_width=True)