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
- a lightweight Chroma-compatible embedding class designed for Render's free 512 MB tier

## Dataset

The default searchable corpus contains 10 primary easy Japanese news articles. Two extra articles were collected as backups but are not included in the default app corpus.

The article texts live in [`articles/`](./articles), and metadata is tracked in [`articles/source-log-template.md`](./articles/source-log-template.md).

## Retrieval Setup

- Embedding setup: lightweight deterministic multilingual text embeddings
- Vector store: in-memory Chroma index
- Default visible results: top 3 unique articles
- Current default chunking strategy: `A` (`160/30`)
- Chunking strategies:
  - `A`: `160/30`
  - `B`: `280/50`
  - `C`: `420/70`

The app builds only the selected tiny Chroma index when the Search page is used, which keeps memory usage low for Render.

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

- The first search builds a small in-memory index, but it should be much lighter than loading a transformer model.
- Generated local artifacts such as `.cache/` and `chroma_db/` are ignored by git.
