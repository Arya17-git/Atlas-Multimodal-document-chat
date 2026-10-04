from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from multimodal_rag.engine import (
    DEFAULT_OLLAMA_MODEL,
    RetrievedSource,
    ask_ollama,
    ingest_pdf,
    load_embedding_models,
    open_vector_store,
    retrieve,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / ".data"
IMAGE_DIRECTORY = DATA_DIRECTORY / "page-images"
MAX_PDF_SIZE = 50 * 1024 * 1024


@st.cache_resource(show_spinner=False)
def get_store(path: str) -> tuple[Any, Any, Any]:
    return open_vector_store(path)


@st.cache_resource(show_spinner=False)
def get_models() -> Any:
    return load_embedding_models()


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"Sources · {len(sources)} pages", icon=":material/source:"):
        for source in sources:
            st.markdown(
                f"**[{source['source_id']}] {source['filename']} · "
                f"page {source['page']}**"
            )
            if source["text"]:
                st.caption(source["text"][:900])
            if source["image_path"] and Path(source["image_path"]).is_file():
                st.image(
                    source["image_path"],
                    caption=f"{source['filename']} · page {source['page']}",
                    alt=f"Retrieved page {source['page']} from {source['filename']}",
                    width="stretch",
                )


st.set_page_config(
    page_title="Atlas · Multimodal document chat",
    page_icon=":material/auto_awesome:",
    layout="wide",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.title("Atlas")
    st.caption("A local, multimodal document assistant")
    st.divider()
    st.subheader("Add documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text and a visual snapshot of each page are indexed locally.",
    )
    index_clicked = st.button(
        "Index selected PDFs",
        type="primary",
        icon=":material/library_add:",
        disabled=not uploaded_files,
        width="stretch",
    )
    st.divider()
    st.subheader("Vision model")
    ollama_url = st.text_input(
        "Ollama URL",
        value="http://localhost:11434",
        help="Address of your local or private Ollama server.",
    ).strip()
    ollama_model = st.text_input(
        "Model name",
        value=DEFAULT_OLLAMA_MODEL,
        help="For example: qwen2.5vl:3b",
    ).strip()
    st.caption("Text and image embeddings run locally. Indexed files stay on disk.")
    st.divider()
    if st.button("Clear conversation", icon=":material/delete_sweep:"):
        st.session_state.chat_history = []
        st.rerun()

st.title("Ask your documents")
st.markdown(
    "Search **what a document says** and **what its pages show**. "
    "Answers are grounded in retrieved PDF pages."
)

try:
    _client, text_collection, image_collection = get_store(
        str(DATA_DIRECTORY / "chroma")
    )
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

text_count = text_collection.count()
image_count = image_collection.count()
metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Text passages", f"{text_count:,}")
metric_b.metric("Visual pages", f"{image_count:,}")
metric_c.metric("Vision model", ollama_model or "Not set")

if index_clicked:
    valid_uploads = []
    for uploaded_file in uploaded_files:
        if uploaded_file.size > MAX_PDF_SIZE:
            st.error(
                f"{uploaded_file.name} is larger than the 50 MB per-file limit."
            )
        else:
            valid_uploads.append(uploaded_file)

    if valid_uploads:
        try:
            models = get_models()
            with st.status("Indexing PDFs", expanded=True) as status:
                indexed = []
                progress = st.progress(0.0)
                for index, uploaded_file in enumerate(valid_uploads, start=1):
                    result = ingest_pdf(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        models,
                        text_collection,
                        image_collection,
                        IMAGE_DIRECTORY,
                    )
                    indexed.append(result)
                    st.write(
                        f"**{uploaded_file.name}** — {result['pages']} pages, "
                        f"{result['text_chunks']} text passages"
                    )
                    progress.progress(index / len(valid_uploads))
                status.update(label="PDF indexing complete", state="complete")
            for result in indexed:
                st.toast(
                    f"Indexed {result['pages']} pages from "
                    f"{result['filename']}.",
                    icon=":material/check_circle:",
                )
            st.rerun()
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))

if text_count == 0 and image_count == 0:
    with st.container(border=True):
        st.subheader("Start with a PDF")
        st.write(
            "Upload a manual, report, or research paper in the sidebar. Atlas "
            "indexes its text and rendered pages for retrieval."
        )
        st.caption(
            "Tip: diagrams, charts, tables, and other page visuals can be used "
            "alongside extracted text."
        )

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

prompt = st.chat_input(
    "Ask about a passage, chart, or diagram",
    disabled=(text_count == 0 and image_count == 0),
)
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.status("Searching text and page visuals", expanded=False) as status:
                models = get_models()
                sources: list[RetrievedSource] = retrieve(
                    prompt, models, text_collection, image_collection, top_k=5
                )
                status.update(label="Preparing a document-grounded answer")
                answer = ask_ollama(
                    prompt,
                    sources,
                    base_url=ollama_url,
                    model_name=ollama_model,
                )
                status.update(label="Answer ready", state="complete")
            st.markdown(answer)
            source_data = [
                {
                    "source_id": source.source_id,
                    "filename": source.filename,
                    "page": source.page,
                    "text": source.text,
                    "image_path": source.image_path,
                }
                for source in sources
            ]
            render_sources(source_data)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": source_data}
        )
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
