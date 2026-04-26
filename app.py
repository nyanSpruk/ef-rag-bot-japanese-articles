from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_TITLE = "Easy Japanese News Explorer"
APP_SUBTITLE = (
    "A semantic search tool for reading and reviewing easy Japanese news articles"
)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RESULT_COUNT = 3
INTERNAL_RETRIEVAL_K = 9
DEFAULT_CHUNKING_STRATEGY = "A"
CHUNKING_STRATEGIES = {
    "A": {"chunk_size": 160, "chunk_overlap": 30, "label": "Precision-first"},
    "B": {"chunk_size": 280, "chunk_overlap": 50, "label": "Context-balanced"},
    "C": {"chunk_size": 420, "chunk_overlap": 70, "label": "Context-heavy"},
}

BASE_DIR = Path(__file__).resolve().parent
ARTICLES_DIR = BASE_DIR / "articles"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db")))
MODEL_CACHE_DIR = Path(
    os.getenv("HF_HOME", str(BASE_DIR / ".cache" / "huggingface"))
)
IS_RENDER = bool(os.getenv("RENDER")) or "render.com" in os.getenv("RENDER_EXTERNAL_URL", "")
RENDER_SAFE_STRATEGIES = [DEFAULT_CHUNKING_STRATEGY]

PRIMARY_ARTICLES = {
    "article-01": {
        "title": "中国の古いお寺で火事がありました",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/6c82515ee530a32703ae0a62c8fb6e3a",
        "publication_date": "2025-11-14",
        "retrieval_date": "2026-04-26",
        "topic_tag": "history-safety",
        "language": "ja",
        "summary_note": "Temple fire in China involving a historically important site.",
    },
    "article-03": {
        "title": "海や川の水が温かくなって魚やのりがとれなくなった",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/b562bf664dbafc4586f4dcabba2a7a59",
        "publication_date": "2025-10-09",
        "retrieval_date": "2026-04-26",
        "topic_tag": "environment-fisheries",
        "language": "ja",
        "summary_note": "Warmer water is affecting fish and seaweed production.",
    },
    "article-04": {
        "title": "マクドナルド、ストローなしの新しいフタを使います",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/2774fba056c4a332a583541b8cad137e",
        "publication_date": "2025-10-29",
        "retrieval_date": "2026-04-26",
        "topic_tag": "business-sustainability",
        "language": "ja",
        "summary_note": "McDonald's is introducing a lid that reduces straw use and plastic waste.",
    },
    "article-05": {
        "title": "ホホジロザメに大きなかみあとが見つかる",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/0733a42e3274ec4e7208fd3a12094ff2",
        "publication_date": "2026-04-25",
        "retrieval_date": "2026-04-26",
        "topic_tag": "wildlife-marine",
        "language": "ja",
        "summary_note": "A great white shark was photographed with a large bite mark.",
    },
    "article-06": {
        "title": "2025年、日本でベトナム人の在留資格が一番多く取り消される",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/fea3ad70549b89332117699c0232d6e0",
        "publication_date": "2026-04-24",
        "retrieval_date": "2026-04-26",
        "topic_tag": "immigration-policy",
        "language": "ja",
        "summary_note": "Japan canceled many residence statuses, especially among Vietnamese nationals.",
    },
    "article-07": {
        "title": "春になると人気が出る桜の味",
        "source_name": "Todaii",
        "source_url": "https://japanese.todaiinews.com/en/news/b9892f7710beb2c09b14afac50c93d03",
        "publication_date": "2026-04-03",
        "retrieval_date": "2026-04-26",
        "topic_tag": "culture-food",
        "language": "ja",
        "summary_note": "Sakura flavors become popular in spring because of seasonality and taste.",
    },
    "article-09": {
        "title": "トマトの値段がとても高くなりました",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/1c93bb5c9daae52a540b42b624aff686",
        "publication_date": "2025-12-15",
        "retrieval_date": "2026-04-26",
        "topic_tag": "food-prices-agriculture",
        "language": "ja",
        "summary_note": "Heavy rain and heat affected tomato production and raised prices.",
    },
    "article-10": {
        "title": "NASAのひこうき 車のタイヤが出ずにおなかで着陸 2人は無事",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/d46aacabfdc3017df5afe894dd859996",
        "publication_date": "2026-01-31",
        "retrieval_date": "2026-04-26",
        "topic_tag": "science-aviation",
        "language": "ja",
        "summary_note": "A NASA plane landed without deploying its wheels, but both passengers were safe.",
    },
    "article-11": {
        "title": "日本に来る外国人が変わりました",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/4c3cfda4c1b09b647cef01f2830191a8",
        "publication_date": "2026-01-30",
        "retrieval_date": "2026-04-26",
        "topic_tag": "tourism-society",
        "language": "ja",
        "summary_note": "The mix of foreign visitors to Japan is changing and travel destinations are broadening.",
    },
    "article-12": {
        "title": "気象庁「今年の夏も暑くなりそう」",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/6515f5c2bb438bfd351f583d030b5687",
        "publication_date": "2025-05-22",
        "retrieval_date": "2026-04-26",
        "topic_tag": "weather-climate",
        "language": "ja",
        "summary_note": "Japan's weather agency expects another hot summer with heavy seasonal rain.",
    },
}

SAMPLE_QUERIES = [
    "トマトの値段はどうして高くなりましたか",
    "日本に来る外国人について知りたい",
    "weather forecast",
    "plastic reduction",
]

CHUNKING_COMPARISON_EXAMPLES = [
    {
        "query": "トマトの値段はどうして高くなりましたか",
        "best_strategy": "A",
        "why": "Strategy A kept the tomato price explanation most focused and direct.",
        "observed": {
            "A": "Returned the tomato article first with the most targeted chunk.",
            "B": "Returned the correct article first, but with broader context.",
            "C": "Returned the correct article first, but with the least precise chunk of the three.",
        },
    },
    {
        "query": "immigration status in Japan",
        "best_strategy": "A",
        "why": "Strategy A ranked the immigration article first, while B and C ranked the tourism article above it.",
        "observed": {
            "A": "Top result was the immigration-policy article.",
            "B": "Top result shifted to the tourism article, with immigration second.",
            "C": "Also preferred the tourism article first, which made the English query weaker.",
        },
    },
]


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7f3ec;
            --panel: rgba(255, 252, 245, 0.97);
            --panel-strong: #fff7eb;
            --border: #dccfb9;
            --text: #182737;
            --text-soft: #45586a;
            --text-muted: #5b6b79;
            --accent: #8a4b35;
            --accent-soft: #efe2d4;
            --link: #8f3d27;
        }
        .stApp {
            background: linear-gradient(180deg, #f3eee5 0%, #fbf9f4 100%);
            color: var(--text);
        }
        .block-container {
            padding-top: 2.35rem;
            padding-bottom: 2.5rem;
            max-width: 1100px;
        }
        div[data-testid="stVerticalBlock"] > div:has(> .hero-card),
        div[data-testid="stVerticalBlock"] > div:has(> .section-card),
        div[data-testid="stVerticalBlock"] > div:has(> .result-card) {
            margin-bottom: 1rem;
        }
        .hero-card, .section-card, .result-card, .query-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 10px 30px rgba(22, 37, 53, 0.06);
            overflow: hidden;
        }
        .hero-card {
            background: linear-gradient(135deg, #fff8ee 0%, #f7efdf 100%);
            padding: 1.6rem 1.7rem;
            margin-bottom: 1.1rem;
        }
        .eyebrow {
            display: inline-block;
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .hero-title {
            font-size: 2.55rem;
            line-height: 1.1;
            margin: 0 0 0.4rem 0;
            color: var(--text);
        }
        .hero-subtitle {
            font-size: 1.08rem;
            color: var(--text-soft);
            margin: 0.2rem 0 0.85rem 0;
        }
        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text);
            margin: 0 0 0.65rem 0;
        }
        .muted {
            color: var(--text-muted);
            line-height: 1.65;
        }
        .hero-card p,
        .section-card p,
        .result-card p {
            margin: 0;
        }
        .meta-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin: 0.8rem 0 1rem 0;
        }
        .meta-pill {
            background: var(--accent-soft);
            color: #6d3a2b;
            border-radius: 999px;
            padding: 0.25rem 0.62rem;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .result-text {
            line-height: 1.78;
            font-size: 1rem;
            color: #203243;
            margin: 0;
            white-space: pre-wrap;
        }
        .result-footer {
            margin-top: 1rem;
            padding-top: 0.85rem;
            border-top: 1px solid #eadfcd;
            font-size: 0.95rem;
        }
        .query-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.9rem;
            margin-top: 0.4rem;
        }
        .query-card {
            padding: 0.95rem 1rem;
            min-height: 92px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .query-card strong {
            color: var(--text);
            display: block;
            margin-bottom: 0.45rem;
        }
        .query-card code {
            white-space: pre-wrap;
            word-break: break-word;
            color: #7a3321;
            font-size: 0.94rem;
            line-height: 1.5;
        }
        .section-card ul {
            margin: 0.65rem 0 0 1.1rem;
            padding: 0;
            color: var(--text-soft);
        }
        .section-card li {
            margin-bottom: 0.4rem;
            line-height: 1.6;
        }
        .section-card code {
            color: #7a3321;
            background: #f6ede1;
            padding: 0.1rem 0.3rem;
            border-radius: 6px;
        }
        a, a:visited {
            color: var(--link);
            text-decoration-thickness: 1.5px;
        }
        a:hover {
            color: #71311f;
        }
        [data-testid="stCaptionContainer"] {
            margin: -0.15rem 0 0.9rem 0;
        }
        div[data-testid="stAlert"] {
            border-radius: 16px;
        }
        .stSidebar .stMarkdown {
            color: #1b2e40;
        }
        .stSidebar [data-testid="stRadio"] > div,
        .stSidebar [data-testid="stSelectbox"] > div {
            margin-top: 0.3rem;
        }
        .stTextInput > div > div > input {
            border-radius: 14px;
            border: 1px solid #cfbea6;
            background: #fff9f0;
            color: #182737;
            padding: 0.8rem 0.95rem;
            font-size: 1rem;
            box-shadow: inset 0 1px 2px rgba(24, 39, 55, 0.04);
            caret-color: #8a4b35;
        }
        .stTextInput > div > div > input::placeholder {
            color: #697886;
            opacity: 1;
        }
        .stTextInput > div > div > input:focus {
            border-color: #b97853;
            box-shadow: 0 0 0 1px #b97853;
        }
        .stTextInput label {
            color: var(--text);
            font-weight: 600;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #decfb9;
            border-radius: 18px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_article_text(raw_text: str) -> tuple[str, str]:
    cleaned = raw_text.strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return "", ""

    title_line = lines[0]
    title = title_line.removeprefix("#").strip()
    body = "\n".join(lines[1:]).strip()
    body = re.sub(r"\n{2,}", "\n", body)
    return title, body


def read_article_markdown(article_id: str) -> tuple[str, str]:
    article_path = ARTICLES_DIR / f"{article_id}.md"
    raw_text = article_path.read_text(encoding="utf-8")
    return normalize_article_text(raw_text)


def load_primary_article_metadata() -> dict[str, dict[str, str]]:
    return PRIMARY_ARTICLES


def build_documents() -> list[dict[str, str | int]]:
    documents: list[dict[str, str | int]] = []
    for article_id, metadata in load_primary_article_metadata().items():
        detected_title, body = read_article_markdown(article_id)
        title = metadata["title"] if metadata["title"] else detected_title
        documents.append(
            {
                "id": article_id,
                "title": title,
                "text": body,
                "source_name": metadata["source_name"],
                "source_url": metadata["source_url"],
                "publication_date": metadata["publication_date"],
                "retrieval_date": metadata["retrieval_date"],
                "topic_tag": metadata["topic_tag"],
                "language": metadata["language"],
                "estimated_length_chars": len(body),
                "summary_note": metadata["summary_note"],
            }
        )
    return documents


@st.cache_data(show_spinner=False)
def get_documents() -> list[dict[str, str | int]]:
    return build_documents()


def to_langchain_documents(documents: list[dict[str, str | int]]) -> list[Document]:
    langchain_docs: list[Document] = []
    for doc in documents:
        metadata = {key: value for key, value in doc.items() if key != "text"}
        langchain_docs.append(
            Document(page_content=str(doc["text"]), metadata=metadata)
        )
    return langchain_docs


def split_documents(
    langchain_docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )
    split_docs = splitter.split_documents(langchain_docs)
    article_chunk_counts: dict[str, int] = {}

    for doc in split_docs:
        article_id = str(doc.metadata.get("id", ""))
        chunk_index = article_chunk_counts.get(article_id, 0)
        article_chunk_counts[article_id] = chunk_index + 1
        doc.metadata["chunk_index"] = chunk_index
        doc.metadata["chunk_id"] = f"{article_id}-chunk-{chunk_index}"

    return split_docs


def build_chunk_fingerprint(split_docs: list[Document]) -> str:
    digest = hashlib.sha256()
    for doc in split_docs:
        chunk_id = str(doc.metadata.get("chunk_id", ""))
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(doc.page_content.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def create_vector_store(
    collection_name: str, embeddings: HuggingFaceEmbeddings
) -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def vector_store_matches_expected(
    vector_store: Chroma, expected_chunk_ids: list[str], expected_fingerprint: str
) -> bool:
    existing_count = vector_store._collection.count()
    if existing_count != len(expected_chunk_ids):
        return False

    stored = vector_store.get(include=["metadatas"])
    stored_ids = [
        str(metadata.get("chunk_id", ""))
        for metadata in stored.get("metadatas", [])
        if metadata is not None
    ]
    if set(stored_ids) != set(expected_chunk_ids):
        return False

    stored_fingerprint = vector_store._collection.metadata.get("chunk_fingerprint", "")
    return stored_fingerprint == expected_fingerprint


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=str(MODEL_CACHE_DIR),
    )


def get_vector_store(
    chunk_size: int, chunk_overlap: int
) -> Chroma:
    documents = get_documents()
    langchain_docs = to_langchain_documents(documents)
    split_docs = split_documents(langchain_docs, chunk_size, chunk_overlap)
    embeddings = get_embeddings()

    collection_name = f"easy_japanese_news_{chunk_size}_{chunk_overlap}"
    expected_chunk_ids = [str(doc.metadata["chunk_id"]) for doc in split_docs]
    expected_fingerprint = build_chunk_fingerprint(split_docs)
    vector_store = create_vector_store(collection_name, embeddings)

    if not vector_store_matches_expected(
        vector_store, expected_chunk_ids, expected_fingerprint
    ):
        vector_store._client.delete_collection(collection_name)
        vector_store = create_vector_store(collection_name, embeddings)
        vector_store._collection.modify(
            metadata={"chunk_fingerprint": expected_fingerprint}
        )
        vector_store.add_documents(split_docs, ids=expected_chunk_ids)

    return vector_store


@st.cache_resource(show_spinner=True)
def get_cached_vector_store(chunk_size: int, chunk_overlap: int) -> Chroma:
    return get_vector_store(chunk_size, chunk_overlap)


def search_documents(
    vector_store: Chroma,
    query: str,
    k: int = DEFAULT_RESULT_COUNT,
    internal_k: int = INTERNAL_RETRIEVAL_K,
) -> list[Document]:
    raw_results = vector_store.similarity_search(query, k=max(k, internal_k))
    unique_results: list[Document] = []
    seen_article_ids: set[str] = set()

    for result in raw_results:
        article_id = str(result.metadata.get("id", ""))
        if article_id in seen_article_ids:
            continue
        seen_article_ids.add(article_id)
        unique_results.append(result)
        if len(unique_results) >= k:
            break

    return unique_results


def get_available_strategy_keys() -> list[str]:
    if IS_RENDER:
        return RENDER_SAFE_STRATEGIES
    return list(CHUNKING_STRATEGIES.keys())


def render_sidebar() -> tuple[str, str]:
    st.sidebar.markdown(
        """
        <div class="section-card">
            <div class="eyebrow">Study Tool</div>
            <div class="section-title">Easy Japanese News Explorer</div>
            <div class="muted">10 easy Japanese news articles • semantic search • built for language learning</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigate",
        ["Home", "Search", "About the Dataset", "Chunking Comparison"],
        label_visibility="collapsed",
    )

    available_strategy_keys = get_available_strategy_keys()
    default_strategy_index = available_strategy_keys.index(DEFAULT_CHUNKING_STRATEGY)

    strategy_label = st.sidebar.selectbox(
        "Chunking strategy",
        options=available_strategy_keys,
        index=default_strategy_index,
        format_func=lambda key: (
            f"{key} · {CHUNKING_STRATEGIES[key]['label']} "
            f"({CHUNKING_STRATEGIES[key]['chunk_size']}/{CHUNKING_STRATEGIES[key]['chunk_overlap']})"
        ),
        help="Use this to preview retrieval behavior across the planned chunking settings.",
    )

    if IS_RENDER:
        st.sidebar.caption(
            "Render mode uses the validated default strategy only to keep memory use low on a 512MB service."
        )
    else:
        st.sidebar.caption(
            "Model files and vector indexes are cached locally after the first successful run."
        )

    return page, strategy_label


def render_home_page() -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="eyebrow">Japanese Reading Practice</div>
            <h1 class="hero-title">{APP_TITLE}</h1>
            <p class="hero-subtitle">{APP_SUBTITLE}</p>
            <p class="muted">
                This app helps Japanese learners search a small collection of easy news articles by topic,
                event, place, or key idea. Instead of rereading every article manually, you can ask focused
                questions and retrieve the most relevant article chunks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.1, 0.9], gap="large")
    with col1:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">How Semantic Search Works</div>
                <p class="muted">
                    The app breaks each article into smaller chunks, turns them into multilingual embeddings,
                    and searches for the chunks that are semantically closest to your question. That means
                    you can search by meaning, not only by exact keyword matches.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">What You Can Ask</div>
                <p class="muted">
                    Ask about weather, prices, tourism, sharks, sustainability, aviation, public safety,
                    or seasonal culture. Japanese queries should work best, and simple English topic queries
                    are supported as well.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        query_cards = "".join(
            f'<div class="query-card"><strong>Sample query</strong><br><code>{query}</code></div>'
            for query in SAMPLE_QUERIES
        )
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-title">Try These Example Queries</div>
                <div class="query-grid">{query_cards}</div>
                <p class="muted" style="margin-top: 0.9rem;">
                    Open the <strong>Search</strong> page from the sidebar to start exploring the dataset.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def format_result_card(result: Document, rank: int) -> str:
    metadata = result.metadata
    topic_tag = metadata.get("topic_tag", "")
    publication_date = metadata.get("publication_date", "")
    source_name = metadata.get("source_name", "")
    source_url = metadata.get("source_url", "#")
    title = metadata.get("title", "Untitled article")

    return f"""
    <div class="result-card">
        <div class="eyebrow">Result {rank}</div>
        <div class="section-title">{title}</div>
        <div class="meta-row">
            <span class="meta-pill">{topic_tag}</span>
            <span class="meta-pill">{publication_date}</span>
            <span class="meta-pill">{source_name}</span>
        </div>
        <p class="result-text">{result.page_content}</p>
        <div class="result-footer muted">
            <a href="{source_url}" target="_blank">Open source article</a>
        </div>
    </div>
    """


def render_search_page(vector_store: Chroma, strategy_key: str) -> None:
    strategy = CHUNKING_STRATEGIES[strategy_key]
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">Search</div>
            <div class="section-title">Find the most relevant article chunks</div>
            <p class="muted">
                Try a Japanese question for the strongest results, or use a simple English topic query to
                test multilingual retrieval.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Active chunking strategy: {strategy_key} · {strategy['label']} "
        f"({strategy['chunk_size']}/{strategy['chunk_overlap']})"
    )

    query = st.text_input(
        "Ask about the article collection",
        placeholder="トマトの値段はどうして高くなりましたか / weather forecast / plastic reduction",
    )

    if not query.strip():
        st.info(
            "Try asking about weather, tourism, prices, sharks, safety, or seasonal culture."
        )
        return

    results = search_documents(vector_store, query.strip(), k=DEFAULT_RESULT_COUNT)
    if not results:
        st.warning("No results were found. Try a shorter or more topic-specific query.")
        return

    st.markdown(
        """
        <p class="muted">
            Results may be limited because the dataset only contains 10 easy Japanese news articles.
            If the matches look weak, try a more focused query.
        </p>
        """,
        unsafe_allow_html=True,
    )

    for index, result in enumerate(results, start=1):
        st.markdown(format_result_card(result, index), unsafe_allow_html=True)


def render_about_dataset_page(documents: list[dict[str, str | int]]) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Why this corpus works well for language learning</div>
            <p class="muted">
                The selected articles cover weather, tourism, sustainability, wildlife, immigration, food,
                agriculture, aviation, safety, and climate. That breadth makes semantic retrieval more useful
                and gives the project stronger report material for chunking comparisons.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = []
    for doc in documents:
        rows.append(
            {
                "Article": doc["title"],
                "Topic": doc["topic_tag"],
                "Published": doc["publication_date"],
                "Source": doc["source_name"],
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def render_chunking_comparison_page() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">Chunking</div>
            <div class="section-title">How the app compares chunking strategies</div>
            <p class="muted">
                Chunking breaks long text into smaller pieces before the app creates embeddings. Different
                chunk sizes can change whether results feel more precise or more contextual.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Current takeaway</div>
            <p class="muted">
                The app currently uses <strong>Strategy A (160/30)</strong> as the default because it performed
                best in the local validation pass. It gave the strongest precision on focused questions and
                handled the tested English immigration query better than the larger chunk settings.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">What the comparison shows</div>
            <p class="muted">
                Smaller chunks usually helped when the query asked for one specific fact. Larger chunks gave
                more surrounding explanation, but that extra context sometimes made related articles rank
                above the most exact match. This tradeoff is the main reason chunking matters in this app.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for example in CHUNKING_COMPARISON_EXAMPLES:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="eyebrow">Example Query</div>
                <div class="section-title"><code>{example["query"]}</code></div>
                <p class="muted">
                    Best strategy in this case: <strong>{example["best_strategy"]}</strong>.
                    {example["why"]}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            [
                {"Strategy": "A", "Observed result": example["observed"]["A"]},
                {"Strategy": "B", "Observed result": example["observed"]["B"]},
                {"Strategy": "C", "Observed result": example["observed"]["C"]},
            ],
            width="stretch",
            hide_index=True,
        )


def main() -> None:
    configure_page()
    inject_styles()

    page, strategy_key = render_sidebar()

    if page == "Home":
        render_home_page()
    elif page == "Search":
        strategy = CHUNKING_STRATEGIES[strategy_key]
        vector_store = get_cached_vector_store(
            chunk_size=int(strategy["chunk_size"]),
            chunk_overlap=int(strategy["chunk_overlap"]),
        )
        render_search_page(vector_store, strategy_key)
    elif page == "About the Dataset":
        render_about_dataset_page(get_documents())
    else:
        render_chunking_comparison_page()


if __name__ == "__main__":
    main()
