
import hashlib
import re
from datetime import datetime
from functools import lru_cache

import fitz
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuMind | AI PDF Assistant",
    page_icon="📚",
    layout="wide",
)

DEFAULT_MODEL = "qwen2.5:0.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How many relevant text chunks to send to the local LLM.
TOP_K = 5


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_pdf_text(text: str) -> str:
    """Normalize PDF text while preserving meaningful punctuation."""

    if not text:
        return ""

    # Remove soft hyphens.
    text = text.replace("\u00ad", "")

    # Join words split by a hyphen at the end of a PDF line.
    # Example: educa-\ntional -> educational
    text = re.sub(
        r"(?<=\w)-\s*\n\s*(?=\w)",
        "",
        text,
    )

    # Convert remaining line breaks and tabs into spaces.
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Normalize repeated spaces.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_for_matching(text: str) -> str:
    """Prepare text for tolerant keyword matching."""

    text = normalize_pdf_text(text).lower()

    # Normalize several common hyphenated forms.
    text = re.sub(r"(?<=\w)[‐‑‒–—-](?=\w)", " ", text)

    # Remove punctuation differences for matching.
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pages(file_bytes: bytes) -> list[dict]:
    """Extract text from each page of an uploaded PDF."""

    pages = []

    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            raw_text = page.get_text("text")
            normalized_text = normalize_pdf_text(raw_text)

            pages.append(
                {
                    "page": page_number,
                    "text": normalized_text,
                }
            )

    return pages


def split_into_sentences(text: str) -> list[str]:
    """Split normalized text into approximate sentences."""

    text = normalize_pdf_text(text)

    if not text:
        return []

    # A practical sentence splitter for ordinary PDF documents.
    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# SEMANTIC SEARCH
# ============================================================

@st.cache_resource(show_spinner="Loading the text-search model...")
def load_embedding_model():
    """Load the sentence-transformer model once."""

    return SentenceTransformer(EMBEDDING_MODEL)


def create_chunks(
    pages: list[dict],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[dict]:
    """Split PDF text into overlapping chunks."""

    chunks = []

    for page_data in pages:
        text = page_data["text"]

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "page": page_data["page"],
                        "text": chunk_text,
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks


def build_index(file_bytes: bytes) -> dict:
    """Extract PDF text and create a semantic-search index."""

    document_hash = hashlib.sha256(file_bytes).hexdigest()
    pages = extract_pages(file_bytes)
    chunks = create_chunks(pages)

    index = {
        "document_hash": document_hash,
        "pages": pages,
        "chunks": chunks,
        "embeddings": None,
        "embedding_error": None,
    }

    if not chunks:
        return index

    # Semantic search is useful, but keyword search remains available
    # if the embedding model cannot load.
    try:
        model = load_embedding_model()

        texts = [chunk["text"] for chunk in chunks]

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        index["embeddings"] = np.asarray(
            embeddings,
            dtype=np.float32,
        )

    except Exception as exc:
        index["embedding_error"] = str(exc)

    return index


def keyword_score(question: str, text: str) -> float:
    """Calculate simple keyword overlap."""

    question_words = set(
        re.findall(r"\b[a-z0-9]+\b", question.lower())
    )

    text_words = set(
        re.findall(r"\b[a-z0-9]+\b", text.lower())
    )

    # Ignore very common words.
    stop_words = {
        "what", "when", "where", "which", "who", "whom",
        "will", "would", "could", "should", "does", "did",
        "the", "a", "an", "is", "are", "was", "were",
        "of", "to", "in", "on", "for", "and", "or",
        "tell", "me", "about", "please", "this", "that",
        "it", "they", "their", "its", "be", "by", "from",
    }

    question_words -= stop_words

    if not question_words:
        return 0.0

    overlap = len(question_words & text_words)

    return overlap / len(question_words)


def retrieve(
    question: str,
    index: dict,
    top_k: int = TOP_K,
) -> list[dict]:
    """Retrieve relevant chunks using semantic and keyword scores."""

    chunks = index["chunks"]

    if not chunks:
        return []

    embeddings = index.get("embeddings")

    if embeddings is not None:
        try:
            model = load_embedding_model()

            question_embedding = model.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]

            semantic_scores = embeddings @ question_embedding

        except Exception:
            semantic_scores = np.zeros(len(chunks))

    else:
        semantic_scores = np.zeros(len(chunks))

    results = []

    for position, chunk in enumerate(chunks):
        semantic = float(semantic_scores[position])
        keyword = keyword_score(question, chunk["text"])

        # Keyword search still works if embeddings are unavailable.
        combined_score = (0.75 * semantic) + (0.25 * keyword)

        results.append(
            {
                **chunk,
                "score": combined_score,
                "semantic_score": semantic,
                "keyword_score": keyword,
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]


# ============================================================
# LAUNCH-DATE EVIDENCE DETECTION
# ============================================================

# Accept several ways of describing an educational AI assistant.
# This intentionally allows variations in the wording.
PROJECT_PATTERN = re.compile(
    r"""
    \b(?:
        educational\s+(?:ai\s+)?assistant |
        education\s+(?:ai\s+)?assistant |
        ai\s+assistant\s+(?:for|in)\s+education |
        assistant\s+for\s+educational\s+(?:use|purposes) |
        ai[- ]based\s+educational\s+assistant |
        educational\s+ai\s+tool |
        education\s+ai\s+tool
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

LAUNCH_PATTERN = re.compile(
    r"""
    \b(?:
        launch(?:es|ed|ing)? |
        release(?:s|d)? |
        rollout |
        roll\s+out |
        go(?:es|ing)?\s+live |
        introduce(?:s|d|ing)? |
        debut(?:s|ed)? |
        publish(?:es|ed)? |
        available |
        planned |
        scheduled |
        expected |
        target(?:ed|ing)? |
        launch\s+date |
        release\s+date
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

MONTH_PATTERN = re.compile(
    r"\b("
    r"January|February|March|April|May|June|July|August|"
    r"September|October|November|December"
    r")\b",
    re.IGNORECASE,
)

YEAR_PATTERN = re.compile(r"\b(?:19|20|21)\d{2}\b")

# Examples:
# October 15, 2027
# 15 October 2027
# 2027-10-15
# 15/10/2027
# 10/15/2027
FULL_DATE_PATTERNS = [
    re.compile(
        r"\b("
        r"January|February|March|April|May|June|July|August|"
        r"September|October|November|December"
        r")\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+((?:19|20|21)\d{2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+("
        r"January|February|March|April|May|June|July|August|"
        r"September|October|November|December"
        r")[,]?\s+((?:19|20|21)\d{2})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:19|20|21)\d{2})-(\d{1,2})-(\d{1,2})\b"
    ),
    re.compile(
        r"\b(\d{1,2})/(\d{1,2})/((?:19|20|21)\d{2})\b"
    ),
]


def extract_full_date(text: str) -> str | None:
    """Find a clearly written full date without inventing one."""

    for pattern in FULL_DATE_PATTERNS:
        match = pattern.search(text)

        if not match:
            continue

        groups = match.groups()

        try:
            if pattern is FULL_DATE_PATTERNS[0]:
                month, day, year = groups
                parsed = datetime.strptime(
                    f"{month} {day} {year}",
                    "%B %d %Y",
                )

            elif pattern is FULL_DATE_PATTERNS[1]:
                day, month, year = groups
                parsed = datetime.strptime(
                    f"{day} {month} {year}",
                    "%d %B %Y",
                )

            elif pattern is FULL_DATE_PATTERNS[2]:
                year, month, day = groups
                parsed = datetime(
                    int(year),
                    int(month),
                    int(day),
                )

            else:
                first, second, year = groups

                # Numeric dates can be ambiguous between DD/MM and MM/DD.
                # Do not silently interpret ambiguous dates.
                first_num = int(first)
                second_num = int(second)

                if first_num <= 12 and second_num <= 12:
                    continue

                if first_num > 12:
                    day, month = first_num, second_num
                else:
                    month, day = first_num, second_num

                parsed = datetime(
                    int(year),
                    month,
                    day,
                )

            return parsed.strftime("%B %d, %Y")

        except ValueError:
            continue

    return None


def find_launch_evidence(index: dict) -> list[dict]:
    """
    Search the actual PDF text for a project mention and related
    launch information. Returns evidence only; it does not guess dates.
    """

    evidence = []

    for page_data in index["pages"]:
        text = page_data["text"]

        if not text:
            continue

        sentences = split_into_sentences(text)

        # Each project mention gets its own local context.
        for i, sentence in enumerate(sentences):
            if not PROJECT_PATTERN.search(sentence):
                continue

            start = max(0, i - 2)
            end = min(len(sentences), i + 4)

            context_sentences = sentences[start:end]
            context = " ".join(context_sentences)

            if LAUNCH_PATTERN.search(context):
                evidence.append(
                    {
                        "page": page_data["page"],
                        "project_sentence": sentence,
                        "context": context,
                    }
                )

        # Also search around project mentions in the entire page.
        # This can catch sentence-boundary and line-wrap problems.
        for match in PROJECT_PATTERN.finditer(text):
            start = max(0, match.start() - 300)
            end = min(len(text), match.end() + 500)

            context = text[start:end]

            if LAUNCH_PATTERN.search(context):
                item = {
                    "page": page_data["page"],
                    "project_sentence": text[match.start():match.end()],
                    "context": context.strip(),
                }

                # Avoid adding the same context more than once.
                if not any(
                    existing["page"] == item["page"]
                    and existing["context"] == item["context"]
                    for existing in evidence
                ):
                    evidence.append(item)

    return evidence


def answer_launch_question(
    question: str,
    index: dict,
) -> str:
    """Answer launch questions only when the PDF provides evidence."""

    evidence_items = find_launch_evidence(index)

    if not evidence_items:
        return (
            "I could not find enough reliable launch-related evidence "
            "for the educational AI assistant in the extracted PDF text. "
            "Please check the PDF text preview and the launch-evidence "
            "debug panel below. The document may use different wording, "
            "or the relevant text may not be extractable."
        )

    # Prefer the most relevant evidence with a year.
    evidence_items.sort(
        key=lambda item: (
            bool(YEAR_PATTERN.search(item["context"])),
            bool(MONTH_PATTERN.search(item["context"])),
            bool(extract_full_date(item["context"])),
        ),
        reverse=True,
    )

    evidence = evidence_items[0]["context"]

    full_date = extract_full_date(evidence)
    year_match = YEAR_PATTERN.search(evidence)
    month_match = MONTH_PATTERN.search(evidence)

    asks_year = bool(
        re.search(r"\b(year|which year|what year)\b", question, re.I)
    )

    asks_month = bool(
        re.search(r"\b(month|which month|what month)\b", question, re.I)
    )

    asks_weekday = bool(
        re.search(
            r"\b(what day|which day|day of the week|weekday)\b",
            question,
            re.I,
        )
    )

    asks_exact_date = bool(
        re.search(
            r"\b(exact date|full date|exactly when|year month and day|"
            r"specific date)\b",
            question,
            re.I,
        )
    )

    if full_date:
        parsed = datetime.strptime(full_date, "%B %d, %Y")
        weekday = parsed.strftime("%A")

        if asks_weekday:
            return (
                f"The PDF gives the launch date as {full_date}. "
                f"That date falls on a {weekday}."
            )

        if asks_month:
            return (
                f"The launch month is {parsed.strftime('%B')}, "
                f"based on the date stated in the PDF: {full_date}."
            )

        if asks_year:
            return (
                f"The launch year is {parsed.year}, "
                f"based on the date stated in the PDF."
            )

        return f"The PDF states the launch date as {full_date}."

    # No full date: report only the date components actually supported.
    year = year_match.group(0) if year_match else None
    month = month_match.group(0) if month_match else None

    if asks_weekday:
        return (
            "The PDF does not provide a complete launch date, so I "
            "cannot determine the day of the week reliably."
        )

    if asks_month:
        if month:
            return f"The PDF mentions {month} in the launch-related evidence."
        return (
            "The PDF does not specify the launch month. "
            + (f"It mentions the year {year}." if year else "")
        )

    if asks_year:
        if year:
            return f"The launch year mentioned in the PDF is {year}."
        return "The PDF does not specify a launch year in the matching evidence."

    if asks_exact_date:
        parts = []

        if year:
            parts.append(f"year: {year}")

        if month:
            parts.append(f"month: {month}")

        if parts:
            known = "; ".join(parts)
            return (
                f"The PDF provides only partial date information ({known}). "
                "It does not establish a complete launch date or day."
            )

        return (
            "The PDF mentions launch-related information, but I could not "
            "verify a complete launch date."
        )

    if year and month:
        return (
            f"The launch-related evidence mentions {month} {year}, "
            "but it does not establish the exact day."
        )

    if year:
        return (
            f"The launch-related evidence mentions {year}, "
            "but the exact month and day are not specified."
        )

    return (
        "I found launch-related wording, but the matching text does not "
        "provide a verifiable launch date."
    )


# ============================================================
# QUESTION ROUTING
# ============================================================

def is_launch_question(question: str) -> bool:
    """Recognize questions about the educational assistant's launch."""

    question_lower = question.lower()

    launch_words = re.search(
        r"\b(launch|launched|release|released|rollout|go live|"
        r"release date|launch date|year|month|weekday|"
        r"day of the week|exact date)\b",
        question_lower,
    )

    project_words = re.search(
        r"\b(educational|education|assistant|project|release date|"
        r"launch date|launch|it)\b",
        question_lower,
    )

    return bool(launch_words and project_words)


def answer_with_ollama(
    question: str,
    retrieved_chunks: list[dict],
    model_name: str,
) -> str:
    """Generate a grounded answer using the local Ollama model."""

    if not retrieved_chunks:
        return (
            "I could not find relevant text in this PDF to answer "
            "your question."
        )

    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Page {chunk['page']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are DocuMind, a careful PDF question-answering assistant.

Rules:
1. Answer using only the supplied PDF context.
2. Do not invent names, dates, numbers, facts, or explanations.
3. If the answer is missing, say that the PDF does not specify it.
4. If the context contains only a year, do not invent a month or day.
5. Keep the answer clear and concise.
6. You may say that information is missing; never pretend it is present.

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                },
            },
            timeout=180,
        )

        response.raise_for_status()
        data = response.json()

        answer = data.get("response", "").strip()

        if not answer:
            return "The local AI model returned an empty answer."

        return answer

    except requests.exceptions.ConnectionError:
        return (
            "Cannot connect to Ollama. Please open Ollama and check that "
            f"the model '{model_name}' is installed."
        )

    except requests.exceptions.Timeout:
        return (
            "The local AI model took too long to respond. "
            "Try a shorter question or check your computer's resources."
        )

    except requests.exceptions.RequestException as exc:
        return f"Could not get a response from Ollama: {exc}"

    except (ValueError, KeyError):
        return "Ollama returned an unexpected response."


def answer_question(
    question: str,
    index: dict,
    model_name: str,
) -> tuple[str, list[dict]]:
    """Route launch-date questions and ordinary PDF questions."""

    if is_launch_question(question):
        answer = answer_launch_question(question, index)
        return answer, []

    retrieved_chunks = retrieve(question, index)

    answer = answer_with_ollama(
        question,
        retrieved_chunks,
        model_name,
    )

    return answer, retrieved_chunks


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title("📚 DocuMind")
st.caption("Ask questions about your PDF using local AI.")

with st.sidebar:
    st.header("Settings")

    model_name = st.text_input(
        "Ollama model",
        value=DEFAULT_MODEL,
        help="Enter a model name shown by the 'ollama list' command.",
    )

    st.markdown("---")

    st.markdown("**Local AI API**")
    st.code(OLLAMA_URL, language="text")

    st.caption(
        "Your PDF is processed by this app. Answers from the language "
        "model use your local Ollama installation."
    )

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Choose a text-based PDF for best results.",
)

if uploaded_file is None:
    st.info("Upload a PDF to get started.")

    st.markdown(
        """
        **How to use DocuMind**

        1. Upload a PDF.
        2. Wait for the text-search index to be created.
        3. Enter a question.
        4. Review the answer and supporting PDF pages.
        """
    )

    st.stop()


# ------------------------------------------------------------
# BUILD INDEX
# ------------------------------------------------------------

file_bytes = uploaded_file.getvalue()
file_hash = hashlib.sha256(file_bytes).hexdigest()

if (
    "documind_index" not in st.session_state
    or st.session_state.get("documind_file_hash") != file_hash
):
    try:
        with st.spinner("Extracting PDF text and preparing search..."):
            index = build_index(file_bytes)

        st.session_state["documind_index"] = index
        st.session_state["documind_file_hash"] = file_hash
        st.session_state["documind_history"] = []

    except Exception as exc:
        st.error(f"Could not process this PDF: {exc}")
        st.stop()

index = st.session_state["documind_index"]

if not any(page["text"] for page in index["pages"]):
    st.error(
        "No readable text was extracted from this PDF. "
        "It may be a scanned document that needs OCR."
    )
    st.stop()

st.success(
    f"PDF loaded: {uploaded_file.name} "
    f"({len(index['pages'])} pages, {len(index['chunks'])} text chunks)"
)

if index.get("embedding_error"):
    st.warning(
        "Semantic search could not load its embedding model. "
        "Keyword search is still available. "
        "Check your internet connection if this is the first run."
    )


# ------------------------------------------------------------
# PDF TEXT PREVIEW
# ------------------------------------------------------------

with st.expander("📄 Preview extracted PDF text"):
    page_options = [
        f"Page {page['page']}"
        for page in index["pages"]
    ]

    selected_page_label = st.selectbox(
        "Select a page",
        page_options,
        key="preview_page",
    )

    selected_page_number = int(
        selected_page_label.replace("Page ", "")
    )

    selected_page = next(
        page
        for page in index["pages"]
        if page["page"] == selected_page_number
    )

    st.text_area(
        "Extracted text",
        value=selected_page["text"],
        height=260,
        disabled=True,
    )


# ------------------------------------------------------------
# LAUNCH EVIDENCE DEBUG PANEL
# ------------------------------------------------------------

with st.expander("🔎 Debug launch-date evidence"):
    launch_evidence = find_launch_evidence(index)

    if launch_evidence:
        st.write(
            f"Found {len(launch_evidence)} matching evidence section(s)."
        )

        for number, item in enumerate(launch_evidence, start=1):
            st.markdown(
                f"**Evidence {number} — Page {item['page']}**"
            )
            st.write(item["context"])

    else:
        st.warning(
            "No matching launch evidence was found using the current "
            "project and launch patterns."
        )

        st.write(
            "Check the extracted PDF text above. The document may use "
            "different wording, or the PDF may not contain the expected "
            "launch information."
        )

        relevant_lines = []

        for page in index["pages"]:
            for sentence in split_into_sentences(page["text"]):
                if re.search(
                    r"2027|assistant|launch|educational|education|release",
                    sentence,
                    re.IGNORECASE,
                ):
                    relevant_lines.append(
                        f"Page {page['page']}: {sentence}"
                    )

        if relevant_lines:
            st.markdown("**Potentially relevant extracted text:**")

            for line in relevant_lines[:30]:
                st.write(line)
        else:
            st.write(
                "No sentences containing the usual education, assistant, "
                "launch, release, or 2027 keywords were found."
            )


# ------------------------------------------------------------
# QUESTION INPUT
# ------------------------------------------------------------

st.markdown("### Ask your PDF")

with st.form("question_form", clear_on_submit=True):
    question = st.text_input(
        "Your question",
        placeholder="Example: What year will the educational AI assistant launch?",
    )

    submitted = st.form_submit_button("Ask DocuMind")

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")

    elif not model_name.strip():
        st.warning("Please enter an Ollama model name.")

    else:
        with st.spinner("Finding an answer..."):
            answer, sources = answer_question(
                question.strip(),
                index,
                model_name.strip(),
            )

        st.session_state.setdefault("documind_history", [])

        st.session_state["documind_history"].append(
            {
                "question": question.strip(),
                "answer": answer,
                "sources": sources,
            }
        )


# ------------------------------------------------------------
# DISPLAY ANSWERS
# ------------------------------------------------------------

history = st.session_state.get("documind_history", [])

if history:
    st.markdown("---")
    st.markdown("### Conversation")

    for item in reversed(history):
        with st.container(border=True):
            st.markdown("**You asked:**")
            st.write(item["question"])

            st.markdown("**DocuMind answered:**")
            st.write(item["answer"])

            sources = item.get("sources", [])

            if sources:
                with st.expander("📌 View supporting PDF passages"):
                    for source_number, source in enumerate(
                        sources,
                        start=1,
                    ):
                        st.markdown(
                            f"**Source {source_number} — "
                            f"Page {source['page']}**"
                        )
                        st.write(source["text"])

    if st.button("Clear conversation"):
        st.session_state["documind_history"] = []
        st.rerun()
