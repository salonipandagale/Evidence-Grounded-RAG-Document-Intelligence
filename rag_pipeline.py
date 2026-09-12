import os
import re
from dataclasses import dataclass
from io import BytesIO

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# MODEL CONFIGURATION
# ============================================================

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"


# ============================================================
# RAG CONFIGURATION
# ============================================================

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP_SENTENCES = 1

# Lower threshold makes the system more flexible for
# different types of user-uploaded documents.
DEFAULT_SCORE_THRESHOLD = 0.25

DEFAULT_TOP_K = 3
DEFAULT_MAX_NEW_TOKENS = 100


# ============================================================
# CHUNK DATA STRUCTURE
# ============================================================

@dataclass
class Chunk:
    chunk_id: int
    doc_id: int
    source: str
    text: str


# ============================================================
# DOCUMENT READING - LOCAL FILES
# ============================================================

def read_txt(path):
    """
    Read a UTF-8 text file from disk.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        return f.read()


def read_pdf(path):
    """
    Extract text from a PDF file on disk.
    """

    reader = PdfReader(path)

    pages = []

    for page in reader.pages:
        pages.append(
            page.extract_text() or ""
        )

    return "\n".join(pages)


# ============================================================
# DOCUMENT READING - STREAMLIT UPLOADS
# ============================================================

def read_uploaded_txt(uploaded_file):
    """
    Read a TXT file uploaded through Streamlit.
    """

    raw_bytes = uploaded_file.getvalue()

    return raw_bytes.decode(
        "utf-8",
        errors="ignore"
    )


def read_uploaded_pdf(uploaded_file):
    """
    Extract text from a PDF uploaded through Streamlit.

    The uploaded file is read directly from memory.
    No permanent copy is required.
    """

    raw_bytes = uploaded_file.getvalue()

    reader = PdfReader(
        BytesIO(raw_bytes)
    )

    pages = []

    for page in reader.pages:
        pages.append(
            page.extract_text() or ""
        )

    return "\n".join(pages)


# ============================================================
# LOAD DOCUMENTS FROM BUILT-IN FOLDER
# ============================================================

def load_documents_from_folder(folder):
    """
    Load all readable TXT and PDF files from a local folder.

    This is used only for built-in/demo documents.
    User uploads do not need to be placed in this folder.
    """

    documents = []

    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Documents folder not found: {folder}"
        )

    for root, _, files in os.walk(folder):

        for filename in sorted(files):

            path = os.path.join(
                root,
                filename
            )

            # ------------------------------------------------
            # TXT
            # ------------------------------------------------

            if filename.lower().endswith(".txt"):

                text = read_txt(path)

            # ------------------------------------------------
            # PDF
            # ------------------------------------------------

            elif filename.lower().endswith(".pdf"):

                text = read_pdf(path)

            else:

                continue

            # ------------------------------------------------
            # Skip empty documents
            # ------------------------------------------------

            if not text.strip():
                continue

            documents.append(
                {
                    "id": len(documents),
                    "source": path,
                    "text": text
                }
            )

    if not documents:
        raise ValueError(
            f"No readable TXT or PDF documents found in: {folder}"
        )

    return documents


# ============================================================
# LOAD DOCUMENTS FROM STREAMLIT UPLOADS
# ============================================================

def load_documents_from_uploaded_files(uploaded_files):
    """
    Convert Streamlit uploaded PDF/TXT files into the
    internal document structure used by the RAG system.

    Supports multiple uploaded files.
    """

    documents = []

    if not uploaded_files:
        raise ValueError(
            "No files were uploaded."
        )

    for uploaded_file in uploaded_files:

        filename = uploaded_file.name

        # ----------------------------------------------------
        # TXT
        # ----------------------------------------------------

        if filename.lower().endswith(".txt"):

            text = read_uploaded_txt(
                uploaded_file
            )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        elif filename.lower().endswith(".pdf"):

            text = read_uploaded_pdf(
                uploaded_file
            )

        else:

            continue

        # ----------------------------------------------------
        # Skip empty files
        # ----------------------------------------------------

        if not text.strip():
            continue

        documents.append(
            {
                "id": len(documents),
                "source": filename,
                "text": text
            }
        )

    if not documents:
        raise ValueError(
            "No readable PDF or TXT documents were uploaded."
        )

    return documents


# ============================================================
# SENTENCE-AWARE CHUNKING
# ============================================================

def sentence_chunk(
    text,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap_sentences=DEFAULT_OVERLAP_SENTENCES
):
    """
    Split text into sentence-aware chunks.

    Parameters
    ----------
    text : str
        Input document text.

    chunk_size : int
        Approximate maximum character size of a chunk.

    overlap_sentences : int
        Number of previous sentences carried into
        the next chunk.

    Returns
    -------
    list[str]
        Sentence-aware document chunks.
    """

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:
        return []

    # --------------------------------------------------------
    # Split at sentence boundaries
    # --------------------------------------------------------

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []

    current_sentences = []
    current_length = 0

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # ----------------------------------------------------
        # If adding this sentence exceeds chunk size,
        # finalize current chunk.
        # ----------------------------------------------------

        if (
            current_sentences
            and current_length + len(sentence) > chunk_size
        ):

            chunks.append(
                " ".join(current_sentences)
            )

            # ------------------------------------------------
            # Preserve sentence overlap
            # ------------------------------------------------

            if overlap_sentences > 0:

                current_sentences = (
                    current_sentences[
                        -overlap_sentences:
                    ]
                )

            else:

                current_sentences = []

            current_length = sum(
                len(s)
                for s in current_sentences
            )

        current_sentences.append(
            sentence
        )

        current_length += (
            len(sentence) + 1
        )

    # --------------------------------------------------------
    # Add final chunk
    # --------------------------------------------------------

    if current_sentences:

        chunks.append(
            " ".join(current_sentences)
        )

    return chunks


# ============================================================
# CREATE CHUNK OBJECTS
# ============================================================

def make_chunks(
    documents,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap_sentences=DEFAULT_OVERLAP_SENTENCES
):
    """
    Convert documents into Chunk objects.
    """

    chunks = []

    for doc in documents:

        text_chunks = sentence_chunk(
            doc["text"],
            chunk_size=chunk_size,
            overlap_sentences=overlap_sentences
        )

        for text in text_chunks:

            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    doc_id=doc["id"],
                    source=doc["source"],
                    text=text
                )
            )

    if not chunks:
        raise ValueError(
            "No chunks could be created from the documents."
        )

    return chunks


# ============================================================
# FAISS VECTOR INDEX
# ============================================================

class RagIndex:

    def __init__(
        self,
        chunks,
        embedder
    ):
        """
        Build a FAISS vector index over document chunks.

        Embeddings are L2-normalized, therefore inner product
        corresponds to cosine similarity.
        """

        if not chunks:
            raise ValueError(
                "Cannot build FAISS index with no chunks."
            )

        self.chunks = chunks
        self.embedder = embedder

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        texts = [
            chunk.text
            for chunk in chunks
        ]

        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")

        # ----------------------------------------------------
        # Normalize embeddings
        # ----------------------------------------------------

        faiss.normalize_L2(
            embeddings
        )

        # ----------------------------------------------------
        # Create FAISS index
        # ----------------------------------------------------

        embedding_dimension = (
            embeddings.shape[1]
        )

        self.index = faiss.IndexFlatIP(
            embedding_dimension
        )

        # ----------------------------------------------------
        # Add vectors
        # ----------------------------------------------------

        self.index.add(
            embeddings
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=DEFAULT_TOP_K,
        score_threshold=DEFAULT_SCORE_THRESHOLD
    ):
        """
        Retrieve the most relevant document chunks.

        Parameters
        ----------
        query : str
            User's question.

        top_k : int
            Maximum number of chunks to retrieve.

        score_threshold : float
            Minimum cosine similarity required.

        Returns
        -------
        list[dict]
            Retrieved chunks with similarity scores.
        """

        if not query or not query.strip():
            return []

        # ----------------------------------------------------
        # Never request more results than vectors in FAISS
        # ----------------------------------------------------

        actual_top_k = min(
            max(1, top_k),
            self.index.ntotal
        )

        # ----------------------------------------------------
        # Embed query
        # ----------------------------------------------------

        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        # ----------------------------------------------------
        # Normalize query
        # ----------------------------------------------------

        faiss.normalize_L2(
            query_embedding
        )

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        scores, indices = self.index.search(
            query_embedding,
            actual_top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx == -1:
                continue

            similarity = float(score)

            # ------------------------------------------------
            # Apply relevance threshold
            # ------------------------------------------------

            if similarity < score_threshold:
                continue

            results.append(
                {
                    "score": similarity,
                    "chunk": self.chunks[idx]
                }
            )

        return results


# ============================================================
# FORMAT RETRIEVED CONTEXT
# ============================================================

def format_context(results):
    """
    Convert retrieved chunks into a context string
    for the language model.
    """

    context_parts = []

    for result in results:

        chunk = result["chunk"]

        source_name = os.path.basename(
            chunk.source
        )

        context_parts.append(
            f"[Source: {source_name} "
            f"| Chunk: {chunk.chunk_id}]\n"
            f"{chunk.text}"
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# RAG PROMPT
# ============================================================

def build_rag_prompt(
    question,
    context
):
    """
    Build a grounded prompt for FLAN-T5.

    The model is instructed to answer only from
    retrieved evidence.
    """

    prompt = f"""
Answer the question using ONLY the context provided below.

If the context does not contain the answer, say:
"I don't know from the provided documents."

Do not use outside knowledge.
Do not invent information.
Keep the answer concise and directly answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt.strip()


# ============================================================
# COMPLETE RAG SYSTEM
# ============================================================

class RAGSystem:

    def __init__(
        self,
        docs_folder
    ):
        """
        Initialize the RAG system.

        Built-in documents are loaded initially.

        Uploaded documents can later replace the active
        knowledge base using load_uploaded_documents().
        """

        self.docs_folder = docs_folder

        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        self.embedder = SentenceTransformer(
            EMBED_MODEL_NAME
        )

        # ----------------------------------------------------
        # Load generation tokenizer
        # ----------------------------------------------------

        self.gen_tokenizer = (
            AutoTokenizer.from_pretrained(
                GEN_MODEL_NAME
            )
        )

        # ----------------------------------------------------
        # Load generation model
        # ----------------------------------------------------

        self.gen_model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                GEN_MODEL_NAME
            )
        )

        # ----------------------------------------------------
        # Load built-in documents
        # ----------------------------------------------------

        self.documents = (
            load_documents_from_folder(
                docs_folder
            )
        )

        # ----------------------------------------------------
        # Create chunks
        # ----------------------------------------------------

        self.chunks = make_chunks(
            self.documents,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap_sentences=DEFAULT_OVERLAP_SENTENCES
        )

        # ----------------------------------------------------
        # Build FAISS index
        # ----------------------------------------------------

        self.rag_index = RagIndex(
            self.chunks,
            self.embedder
        )

    # ========================================================
    # LOAD USER-UPLOADED DOCUMENTS
    # ========================================================

    def load_uploaded_documents(
        self,
        uploaded_files
    ):
        """
        Replace the active knowledge base with
        user-uploaded PDF/TXT documents.

        The uploaded documents are processed dynamically.
        """

        documents = (
            load_documents_from_uploaded_files(
                uploaded_files
            )
        )

        # ----------------------------------------------------
        # Create chunks
        # ----------------------------------------------------

        chunks = make_chunks(
            documents,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap_sentences=DEFAULT_OVERLAP_SENTENCES
        )

        # ----------------------------------------------------
        # Build new FAISS index
        # ----------------------------------------------------

        new_rag_index = RagIndex(
            chunks,
            self.embedder
        )

        # ----------------------------------------------------
        # Replace active knowledge base
        # ----------------------------------------------------

        self.documents = documents
        self.chunks = chunks
        self.rag_index = new_rag_index

        return {
            "documents": self.documents,
            "chunks": self.chunks
        }

    # ========================================================
    # RESTORE BUILT-IN DOCUMENTS
    # ========================================================

    def reset_to_builtin_documents(self):
        """
        Restore the original documents from the
        application's docs folder.
        """

        documents = (
            load_documents_from_folder(
                self.docs_folder
            )
        )

        chunks = make_chunks(
            documents,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap_sentences=DEFAULT_OVERLAP_SENTENCES
        )

        new_rag_index = RagIndex(
            chunks,
            self.embedder
        )

        self.documents = documents
        self.chunks = chunks
        self.rag_index = new_rag_index

        return {
            "documents": self.documents,
            "chunks": self.chunks
        }

    # ========================================================
    # ANSWER QUESTION
    # ========================================================

    def answer(
        self,
        question,
        top_k=DEFAULT_TOP_K,
        score_threshold=DEFAULT_SCORE_THRESHOLD,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS
    ):
        """
        Retrieve relevant evidence and generate
        a grounded answer.
        """

        # ----------------------------------------------------
        # Validate question
        # ----------------------------------------------------

        question = question.strip()

        if not question:

            return {
                "answer": "Please enter a valid question.",
                "results": [],
                "context": ""
            }

        # ----------------------------------------------------
        # Retrieve evidence
        # ----------------------------------------------------

        results = self.rag_index.search(
            question,
            top_k=top_k,
            score_threshold=score_threshold
        )

        # ----------------------------------------------------
        # No relevant evidence
        # ----------------------------------------------------

        if not results:

            return {
                "answer":
                    "I don't know from the provided documents.",
                "results": [],
                "context": ""
            }

        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context = format_context(
            results
        )

        # ----------------------------------------------------
        # Build grounded prompt
        # ----------------------------------------------------

        prompt = build_rag_prompt(
            question,
            context
        )

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        inputs = self.gen_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        outputs = self.gen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        answer = self.gen_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return {
            "answer": answer,
            "results": results,
            "context": context
        }