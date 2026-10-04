# Atlas running guide

This guide walks through installing, starting, and using the local multimodal PDF assistant on Windows, macOS, or Linux.

## Contents

- [Before you begin](#before-you-begin)
- [Install Python dependencies](#install-python-dependencies)
- [Start Ollama](#start-ollama)
- [Run the app](#run-the-app)
- [Index PDFs and ask questions](#index-pdfs-and-ask-questions)
- [Choose a different vision model](#choose-a-different-vision-model)
- [Run the tests](#run-the-tests)
- [Troubleshooting](#troubleshooting)
- [Reset the local index](#reset-the-local-index)

## Before you begin

1. Install Python 3.11 or 3.12, 64-bit.
2. Install [Ollama](https://ollama.com/download).
3. Allow several gigabytes of free disk space for PyTorch, embedding weights, and the Ollama vision model.
4. Open a terminal in the project folder.

> Python 3.14 may not yet have compatible PyTorch wheels for your platform. If dependency installation fails on 3.14, install Python 3.12 and create the environment with `py -3.12`.

## Install Python dependencies

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, use the environment's executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Start Ollama

Install the recommended vision model:

```bash
ollama pull qwen2.5vl:3b
```

Ensure Ollama is running. On Windows and macOS, open the Ollama application. On Linux, start its service or run `ollama serve` in a separate terminal.

Confirm the model is installed:

```bash
ollama list
```

The app defaults to `http://localhost:11434`. If Ollama runs on a different host or port, enter that address in the app sidebar.

## Run the app

Activate the virtual environment if it is not already active, then run:

```bash
streamlit run app.py
```

Streamlit prints a local URL, typically <http://localhost:8501>. Open it in a browser.

## Index PDFs and ask questions

1. In **Add documents** in the sidebar, choose one or more PDF files (up to 50 MB each).
2. Select **Index selected PDFs**.
3. Wait for indexing to finish. The first run downloads BGE and CLIP weights from Hugging Face; later indexing reuses the local cache.
4. Enter a question in the chat box, such as “What does the warning diagram on page 12 mean?”
5. Expand **Sources** under an answer to inspect the retrieved page text and image.
6. Use **Clear conversation** to start a fresh chat. This preserves the document index.

Page text, vector index, and rendered page images are stored in `.data/` under the project directory. That folder is excluded from Git.

## Choose a different vision model

Pull another vision model with Ollama, for example:

```bash
ollama pull qwen2.5vl:7b
```

Enter the installed model's exact name in the **Model name** field in the sidebar. Larger models generally need more memory. Not every Ollama model supports image input; select a vision-language model.

## Run the tests

With the project virtual environment activated:

```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

| Symptom | What to try |
|---|---|
| `Could not connect to Ollama` | Start Ollama, check its URL in the sidebar, and verify that the service is listening on port 11434. |
| Ollama says the model is missing | Run `ollama pull qwen2.5vl:3b` and retry. |
| `pip install` fails on Python 3.14 | Install Python 3.12, remove only this project's `.venv`, and follow the install steps again. |
| The first indexing takes a while | This is expected while models download and page embeddings are generated. Keep the app open until indexing completes. |
| CPU inference is slow | The models use a detected CUDA GPU when available. CPU remains supported; smaller Ollama vision models may respond faster. |
| A scanned PDF has no text results | Its rendered pages can still be visually retrieved. OCR is not included in this starter project. |
| Port 8501 is busy | Run `streamlit run app.py --server.port 8502` and open the URL Streamlit prints. |

## Reset the local index

Stop Streamlit first. To remove indexed PDFs, page images, and the local vector database, delete the project's `.data` folder, then restart the app and index again.

> **Warning:** Removing `.data` permanently deletes the local index and rendered images. It does not delete the original PDF files.
