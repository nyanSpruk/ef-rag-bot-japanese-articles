# Easy Japanese News Explorer

A Streamlit RAG app for searching easy Japanese news articles with semantic search.

## Project Overview

This project builds a small retrieval-augmented search app around 10 easy Japanese news articles collected from Todaii. It is designed for Japanese learners who want to review article content by topic, event, place, or key idea without rereading every article manually.

The app includes:

- `Home`
- `Search`
- `About the Dataset`
- `Chunking Comparison`

It uses:

- Streamlit
- LangChain
- ChromaDB
- sentence-transformers

## Dataset

The default searchable corpus contains 10 primary easy Japanese news articles. Two extra articles were collected as backups but are not included in the default app corpus.

The article texts live in [`articles/`](./articles), and metadata is tracked in [`articles/source-log-template.md`](./articles/source-log-template.md).

## Retrieval Setup

- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Vector store: local persistent `chroma_db/`
- Default visible results: top 3 unique articles
- Current default chunking strategy: `A` (`160/30`)
- Chunking strategies:
  - `A`: `160/30`
  - `B`: `280/50`
  - `C`: `420/70`

The app preloads all 3 chunking strategies on startup and caches Hugging Face model files locally in `.cache/`.

## Run Locally

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

## Deployment

This repo includes a [`render.yaml`](./render.yaml) file for Render deployment.

## Notes

- The first startup can take longer because the app preloads search assets.
- The first search-related model setup may still be heavy on a fresh machine or fresh deployment.
- Generated local artifacts such as `.cache/` and `chroma_db/` are ignored by git.
