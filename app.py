"""Streamlit app for a small RAG-style Japanese news search project.

1. settings and article metadata
2. lightweight embeddings
3. page styling
4. document loading and chunking
5. search logic
6. Streamlit page rendering
"""

from __future__ import annotations

import hashlib
import html
import math
import re
from pathlib import Path
from typing import Iterable

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# App settings
# ---------------------------------------------------------------------------

APP_TITLE = "Easy Japanese News Explorer"
APP_SUBTITLE = (
    "A semantic search tool for reading and reviewing easy Japanese news articles"
)
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

# Render's free tier has very little memory. This keeps the project in the
# required Streamlit + LangChain + Chroma shape without loading a transformer.
EMBEDDING_MODEL_NOTE = "Lightweight hashed multilingual text embeddings for Render"
VECTOR_DIMENSIONS = 768

# These patterns are reused in the embedding and keyword search code.
TOKEN_PATTERN = r"[a-z0-9]+|[一-龯ぁ-んァ-ンー]+"
JAPANESE_PATTERN = r"[一-龯ぁ-んァ-ンー]"
CHUNK_SEPARATORS = ["\n\n", "\n", "。", "、", " ", ""]


# ---------------------------------------------------------------------------
# Article metadata
# ---------------------------------------------------------------------------

PRIMARY_ARTICLES = {
    "article-01": {
        "title": "中国の古いお寺で火事がありました",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/6c82515ee530a32703ae0a62c8fb6e3a",
        "publication_date": "2025-11-14",
        "retrieval_date": "2026-04-26",
        "topic_tag": "history-safety",
        "language": "ja",
        "summary_note": "Temple fire in China involving a historically important site and safety concerns.",
    },
    "article-03": {
        "title": "海や川の水が温かくなって魚やのりがとれなくなった",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/b562bf664dbafc4586f4dcabba2a7a59",
        "publication_date": "2025-10-09",
        "retrieval_date": "2026-04-26",
        "topic_tag": "environment-fisheries",
        "language": "ja",
        "summary_note": "Warmer sea and river water affects fish, seaweed, climate, and fisheries.",
    },
    "article-04": {
        "title": "マクドナルド、ストローなしの新しいフタを使います",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/2774fba056c4a332a583541b8cad137e",
        "publication_date": "2025-10-29",
        "retrieval_date": "2026-04-26",
        "topic_tag": "business-sustainability",
        "language": "ja",
        "summary_note": "McDonald's uses a strawless lid to reduce plastic waste and support sustainability.",
    },
    "article-05": {
        "title": "ホホジロザメに大きなかみあとが見つかる",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/0733a42e3274ec4e7208fd3a12094ff2",
        "publication_date": "2026-04-25",
        "retrieval_date": "2026-04-26",
        "topic_tag": "wildlife-marine",
        "language": "ja",
        "summary_note": "A great white shark was photographed with a large bite mark, likely from another shark.",
    },
    "article-06": {
        "title": "2025年、日本でベトナム人の在留資格が一番多く取り消される",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/fea3ad70549b89332117699c0232d6e0",
        "publication_date": "2026-04-24",
        "retrieval_date": "2026-04-26",
        "topic_tag": "immigration-policy",
        "language": "ja",
        "summary_note": "Japan canceled residence status for foreign residents, especially Vietnamese trainees and students.",
    },
    "article-07": {
        "title": "春になると人気が出る桜の味",
        "source_name": "Todaii",
        "source_url": "https://japanese.todaiinews.com/en/news/b9892f7710beb2c09b14afac50c93d03",
        "publication_date": "2026-04-03",
        "retrieval_date": "2026-04-26",
        "topic_tag": "culture-food",
        "language": "ja",
        "summary_note": "Sakura flavors become popular in spring because they feel seasonal, limited, salty, and special.",
    },
    "article-09": {
        "title": "トマトの値段がとても高くなりました",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/1c93bb5c9daae52a540b42b624aff686",
        "publication_date": "2025-12-15",
        "retrieval_date": "2026-04-26",
        "topic_tag": "food-prices-agriculture",
        "language": "ja",
        "summary_note": "Tomato prices rose because heavy rain, heat, and dry weather hurt agriculture production.",
    },
    "article-10": {
        "title": "NASAのひこうき 車のタイヤが出ずにおなかで着陸 2人は無事",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/d46aacabfdc3017df5afe894dd859996",
        "publication_date": "2026-01-31",
        "retrieval_date": "2026-04-26",
        "topic_tag": "science-aviation",
        "language": "ja",
        "summary_note": "A NASA airplane made an emergency landing without wheels, but both passengers were safe.",
    },
    "article-11": {
        "title": "日本に来る外国人が変わりました",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/4c3cfda4c1b09b647cef01f2830191a8",
        "publication_date": "2026-01-30",
        "retrieval_date": "2026-04-26",
        "topic_tag": "tourism-society",
        "language": "ja",
        "summary_note": "Foreign visitors to Japan are changing, with tourism from Korea, Taiwan, America, and Europe growing.",
    },
    "article-12": {
        "title": "気象庁「今年の夏も暑くなりそう」",
        "source_name": "Todaii",
        "source_url": "https://easyjapanese.net/detail/6515f5c2bb438bfd351f583d030b5687",
        "publication_date": "2025-05-22",
        "retrieval_date": "2026-04-26",
        "topic_tag": "weather-climate",
        "language": "ja",
        "summary_note": "Japan's weather agency forecasts a hot summer, seasonal rain, and heatstroke risk.",
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
            "C": "Returned the correct article first, but with the broadest context.",
        },
    },
    {
        "query": "a strawless",
        "best_strategy": "A",
        "why": "This weak English fragment shows the clearest chunking difference: only Strategy A ranked the strawless-lid article first.",
        "observed": {
            "A": "Top result was the McDonald's article about a strawless lid.",
            "B": "Top result shifted to the NASA airplane article.",
            "C": "Top result also shifted to the NASA airplane article.",
        },
    },
]


# ---------------------------------------------------------------------------
# Lightweight embedding model
# ---------------------------------------------------------------------------


class LightweightMultilingualEmbeddings(Embeddings):
    """Small deterministic embeddings.

    A normal embedding model would download and run a transformer.
    This class creates fixed-size vectors by hashing words and Japanese
    character n-grams. Chroma can still compare the vectors for similarity.
    """

    def __init__(self, dimensions: int = VECTOR_DIMENSIONS) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature, weight in self._features(text):
            index = int(
                hashlib.blake2b(feature.encode("utf-8"), digest_size=4).hexdigest(), 16
            )
            vector[index % self.dimensions] += weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _features(self, text: str) -> Iterable[tuple[str, float]]:
        lowered = text.lower()
        tokens = re.findall(TOKEN_PATTERN, lowered)
        for token in tokens:
            yield token, 2.0
            if re.search(JAPANESE_PATTERN, token):
                for size in (2, 3):
                    for index in range(max(0, len(token) - size + 1)):
                        yield token[index : index + size], 1.0

        compact = re.sub(r"\s+", "", lowered)
        for size in (3, 4, 5):
            for index in range(max(0, len(compact) - size + 1)):
                yield compact[index : index + size], 0.45


# ---------------------------------------------------------------------------
# Streamlit page setup and styling
# ---------------------------------------------------------------------------


def configure_page() -> None:
    """Set Streamlit's browser title, icon, and layout."""

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_styles() -> None:
    """Add the small amount of custom CSS used by all pages."""

    st.markdown(
        """
        <style>
        :root {
            --panel: rgba(255, 252, 245, 0.97);
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
        .hero-card p, .section-card p, .result-card p {
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
        div[data-testid="stAlert"], [data-testid="stDataFrame"] {
            border-radius: 16px;
        }
        .stTextInput > div > div > input {
            border-radius: 14px;
            border: 1px solid #cfbea6;
            background: #fff9f0 !important;
            color: #182737 !important;
            padding: 0.8rem 0.95rem;
            font-size: 1rem;
        }
        div[data-baseweb="input"] {
            background-color: #fff9f0 !important;
            border-radius: 14px;
        }
        .stTextInput label,
        .stTextInput label p,
        div[data-testid="stTextInput"] label,
        div[data-testid="stTextInput"] label p {
            color: #182737 !important;
            font-weight: 700 !important;
        }
        .stTextInput input::placeholder {
            color: #617181 !important;
            opacity: 1 !important;
        }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p {
            color: #45586a !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Document loading and chunking
# ---------------------------------------------------------------------------


def normalize_article_text(raw_text: str) -> tuple[str, str]:
    """Return the Markdown title and body from one article file."""

    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
    if not lines:
        return "", ""
    return lines[0].removeprefix("#").strip(), "\n".join(lines[1:]).strip()


def read_article_markdown(article_id: str) -> tuple[str, str]:
    """Load one article file from the articles folder."""

    return normalize_article_text(
        (ARTICLES_DIR / f"{article_id}.md").read_text(encoding="utf-8")
    )


def build_documents() -> list[dict[str, str | int]]:
    """Combine article text files with their metadata."""

    documents: list[dict[str, str | int]] = []
    for article_id, metadata in PRIMARY_ARTICLES.items():
        detected_title, body = read_article_markdown(article_id)
        title = metadata["title"] or detected_title
        documents.append(
            {
                "id": article_id,
                "title": title,
                "text": body,
                "estimated_length_chars": len(body),
                **metadata,
            }
        )
    return documents


@st.cache_data(show_spinner=False)
def get_documents() -> list[dict[str, str | int]]:
    """Return the article list, cached so Streamlit does not reread files."""

    return build_documents()


def to_langchain_documents(documents: list[dict[str, str | int]]) -> list[Document]:
    """Convert plain dictionaries into LangChain Document objects.

    The page content includes title and English metadata notes as well as the
    Japanese article text. That makes Japanese search work and gives simple
    English topic searches a little useful vocabulary.
    """

    langchain_docs: list[Document] = []
    for doc in documents:
        metadata = {key: value for key, value in doc.items() if key != "text"}
        metadata["display_text"] = str(doc["text"])
        searchable_text = "\n".join(
            [
                str(doc["title"]),
                str(doc["topic_tag"]).replace("-", " "),
                str(doc["summary_note"]),
                str(doc["text"]),
            ]
        )
        langchain_docs.append(Document(page_content=searchable_text, metadata=metadata))
    return langchain_docs


def split_documents(
    langchain_docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split articles into chunks and add stable chunk IDs."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHUNK_SEPARATORS,
    )
    split_docs = splitter.split_documents(langchain_docs)
    chunk_counts: dict[str, int] = {}
    for doc in split_docs:
        article_id = str(doc.metadata["id"])
        chunk_index = chunk_counts.get(article_id, 0)
        chunk_counts[article_id] = chunk_index + 1
        doc.metadata["chunk_index"] = chunk_index
        doc.metadata["chunk_id"] = f"{article_id}-chunk-{chunk_index}"
    return split_docs


def get_split_docs(chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Build chunks for a specific chunking strategy."""

    return split_documents(
        to_langchain_documents(get_documents()), chunk_size, chunk_overlap
    )


@st.cache_resource(show_spinner=False)
def get_cached_split_docs(chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Cache split documents by strategy."""

    return get_split_docs(chunk_size, chunk_overlap)


@st.cache_resource(show_spinner=False)
def get_embeddings() -> LightweightMultilingualEmbeddings:
    """Create the embedding object once per Streamlit process."""

    return LightweightMultilingualEmbeddings()


@st.cache_resource(show_spinner=False)
def get_vector_store(chunk_size: int, chunk_overlap: int) -> Chroma:
    """Create a tiny in-memory Chroma collection for one chunking strategy."""

    split_docs = get_split_docs(chunk_size, chunk_overlap)
    collection_suffix = hashlib.sha1(
        f"{chunk_size}:{chunk_overlap}".encode()
    ).hexdigest()[:8]
    return Chroma.from_documents(
        documents=split_docs,
        embedding=get_embeddings(),
        ids=[str(doc.metadata["chunk_id"]) for doc in split_docs],
        collection_name=f"easy_japanese_news_{collection_suffix}",
    )


# ---------------------------------------------------------------------------
# Search logic
# ---------------------------------------------------------------------------


def extract_query_terms(query: str) -> set[str]:
    """Extract simple English/Japanese terms for the keyword backup search."""

    lowered = query.lower().strip()
    terms = set(re.findall(TOKEN_PATTERN, lowered))
    if re.search(JAPANESE_PATTERN, lowered):
        compact = re.sub(r"\s+", "", lowered)
        terms.update(
            compact[index : index + 2] for index in range(max(0, len(compact) - 1))
        )
    return {term for term in terms if term}


def lexical_search_documents(
    split_docs: list[Document], query: str, limit: int
) -> list[Document]:
    """Return keyword matches used to support the vector search results."""

    query_terms = extract_query_terms(query)
    if not query_terms:
        return []

    scored_docs: list[tuple[float, Document]] = []
    for doc in split_docs:
        fields = {
            "title": str(doc.metadata.get("title", "")).lower(),
            "summary": str(doc.metadata.get("summary_note", "")).lower(),
            "topic": str(doc.metadata.get("topic_tag", "")).replace("-", " ").lower(),
            "body": str(doc.metadata.get("display_text", "")).lower(),
        }
        score = 0.0
        for term in query_terms:
            if term in fields["title"]:
                score += max(6.0, min(len(term) * 2.5, 14.0))
            if term in fields["summary"]:
                score += max(2.5, min(len(term), 6.0))
            if term in fields["topic"]:
                score += max(2.0, min(len(term), 5.0))
            if term in fields["body"]:
                score += max(1.0, min(len(term), 6.0))
        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_docs[:limit]]


def search_documents(
    vector_store: Chroma,
    split_docs: list[Document],
    query: str,
    k: int = DEFAULT_RESULT_COUNT,
    internal_k: int = INTERNAL_RETRIEVAL_K,
) -> list[Document]:
    """Search the corpus and return unique articles.

    Chroma gives semantic matches. The lexical search is a small helper for
    exact article titles, Japanese words, and simple English metadata terms.
    The final loop removes duplicate visible articles from the result list.
    """

    semantic_results = vector_store.similarity_search(query, k=max(k, internal_k))
    lexical_results = lexical_search_documents(
        split_docs, query, limit=max(k, internal_k)
    )
    scores: dict[str, float] = {}
    docs_by_chunk: dict[str, Document] = {}

    for source_weight, results in ((1.0, semantic_results), (1.15, lexical_results)):
        for rank, result in enumerate(results, start=1):
            chunk_id = str(result.metadata["chunk_id"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + source_weight / (rank + 10)
            docs_by_chunk[chunk_id] = result

    unique_results: list[Document] = []
    seen_article_ids: set[str] = set()
    for chunk_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        doc = docs_by_chunk[chunk_id]
        article_id = str(doc.metadata["id"])
        if article_id in seen_article_ids:
            continue
        seen_article_ids.add(article_id)
        unique_results.append(doc)
        if len(unique_results) >= k:
            break

    return unique_results


# ---------------------------------------------------------------------------
# Streamlit page rendering
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[str, str]:
    """Render navigation and return the selected page and chunking strategy."""

    st.sidebar.markdown(
        """
        <div class="section-card">
            <div class="eyebrow">Study Tool</div>
            <div class="section-title">Easy Japanese News Explorer</div>
            <div class="muted">10 easy Japanese news articles • Chroma search • Render-friendly</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigate",
        ["Home", "Search", "About the Dataset", "Chunking Comparison"],
        label_visibility="collapsed",
    )
    strategy = st.sidebar.selectbox(
        "Chunking strategy",
        options=list(CHUNKING_STRATEGIES),
        index=list(CHUNKING_STRATEGIES).index(DEFAULT_CHUNKING_STRATEGY),
        format_func=lambda key: (
            f"{key} · {CHUNKING_STRATEGIES[key]['label']} "
            f"({CHUNKING_STRATEGIES[key]['chunk_size']}/{CHUNKING_STRATEGIES[key]['chunk_overlap']})"
        ),
    )
    return page, strategy


def render_home_page() -> None:
    """Render the landing page."""

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
                    The app breaks each article into smaller chunks, turns those chunks into compact
                    Chroma vectors, and searches for the closest matches. For the small class dataset,
                    this keeps search fast and stable on Render.
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
                    or seasonal culture. Japanese queries work best, and simple English topic queries
                    are supported through English metadata notes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        query_cards = "".join(
            f'<div class="query-card"><strong>Sample query</strong><br><code>{html.escape(query)}</code></div>'
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
    """Create the HTML for one search result card."""

    metadata = result.metadata
    return f"""
    <div class="result-card">
        <div class="eyebrow">Result {rank}</div>
        <div class="section-title">{html.escape(str(metadata.get("title", "Untitled article")))}</div>
        <div class="meta-row">
            <span class="meta-pill">{html.escape(str(metadata.get("topic_tag", "")))}</span>
            <span class="meta-pill">{html.escape(str(metadata.get("publication_date", "")))}</span>
            <span class="meta-pill">{html.escape(str(metadata.get("source_name", "")))}</span>
        </div>
        <p class="result-text">{html.escape(str(metadata.get("display_text", result.page_content)))}</p>
        <div class="result-footer muted">
            <a href="{html.escape(str(metadata.get("source_url", "#")))}" target="_blank">Open source article</a>
        </div>
    </div>
    """


def render_search_page(
    vector_store: Chroma, split_docs: list[Document], strategy_key: str
) -> None:
    """Render the search page and display the current query results."""

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

    results = search_documents(vector_store, split_docs, query.strip())
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
    """Render a small dataset overview table."""

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Why this corpus works well for language learning</div>
            <p class="muted">
                The app uses 10 primary easy Japanese news articles from Todaii. Two backup articles were
                collected but kept outside the default corpus. The selected articles cover weather, tourism,
                sustainability, wildlife, immigration, food, agriculture, aviation, safety, and climate.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    rows = [
        {
            "Article": doc["title"],
            "Topic": doc["topic_tag"],
            "Published": doc["publication_date"],
            "Length": doc["estimated_length_chars"],
            "Source": doc["source_name"],
        }
        for doc in documents
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


def render_chunking_comparison_page() -> None:
    """Render the page that documents the chunking experiments."""

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="eyebrow">Chunking</div>
            <div class="section-title">How the app compares chunking strategies</div>
            <p class="muted">
                Chunking breaks article text into smaller pieces before Chroma stores vectors.
                The current embedding setup is <strong>{EMBEDDING_MODEL_NOTE}</strong>, chosen so the
                deployed app stays inside Render's 512 MB free-tier memory limit.
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
                The app uses <strong>Strategy A (160/30)</strong> by default because validation showed it
                was the most precise for focused factual questions while still supporting simple English
                topic queries through source metadata.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        [
            {
                "Strategy": key,
                "Label": cfg["label"],
                "Chunk size": cfg["chunk_size"],
                "Overlap": cfg["chunk_overlap"],
            }
            for key, cfg in CHUNKING_STRATEGIES.items()
        ],
        width="stretch",
        hide_index=True,
    )
    for example in CHUNKING_COMPARISON_EXAMPLES:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="eyebrow">Example Query</div>
                <div class="section-title"><code>{html.escape(example["query"])}</code></div>
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


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the selected Streamlit page."""

    configure_page()
    inject_styles()
    page, strategy_key = render_sidebar()

    if page == "Home":
        render_home_page()
        return
    if page == "About the Dataset":
        render_about_dataset_page(get_documents())
        return
    if page == "Chunking Comparison":
        render_chunking_comparison_page()
        return

    strategy = CHUNKING_STRATEGIES[strategy_key]
    vector_store = get_vector_store(
        int(strategy["chunk_size"]), int(strategy["chunk_overlap"])
    )
    split_docs = get_cached_split_docs(
        int(strategy["chunk_size"]), int(strategy["chunk_overlap"])
    )
    render_search_page(vector_store, split_docs, strategy_key)


if __name__ == "__main__":
    main()
