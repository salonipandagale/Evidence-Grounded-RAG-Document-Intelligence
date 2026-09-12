import os
import copy
import hashlib

import streamlit as st

from rag_pipeline import RAGSystem


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Evidence-Grounded Document Intelligence",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-top: 10px;
        margin-bottom: 20px;
        background-color: #fafafa;
        font-size: 18px;
        line-height: 1.6;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #c8e6c9;
        background-color: #f1f8f1;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Evidence-Grounded Document Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload documents, ask questions, and inspect the '
    'evidence used to generate each answer.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CONFIGURATION
# ============================================================

DOCS_FOLDER = "docs"

TOP_K = 3
SCORE_THRESHOLD = 0.25


# ============================================================
# LOAD BASE RAG SYSTEM
# ============================================================

@st.cache_resource
def load_base_rag_system():
    """
    Load the base RAG system once.

    The expensive embedding model and language model
    are loaded only once and shared across sessions.

    IMPORTANT:
    This base object is never modified directly.
    """

    return RAGSystem(DOCS_FOLDER)


with st.spinner(
    "Loading embedding model, language model, and document index..."
):

    base_rag = load_base_rag_system()


# ============================================================
# SESSION-SPECIFIC RAG SYSTEM
# ============================================================

if "rag" not in st.session_state:

    # Create a shallow copy of the base system.
    #
    # The models remain shared, while documents,
    # chunks and FAISS index can be replaced for
    # this individual user's session.

    st.session_state.rag = copy.copy(base_rag)


rag = st.session_state.rag


# ============================================================
# SESSION STATE
# ============================================================

if "document_mode" not in st.session_state:

    st.session_state.document_mode = "Built-in Documents"


if "uploaded_signature" not in st.session_state:

    st.session_state.uploaded_signature = None


if "using_uploaded_documents" not in st.session_state:

    st.session_state.using_uploaded_documents = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Document Source")

    document_mode = st.radio(
        "Choose knowledge base",
        [
            "Built-in Documents",
            "Upload Your Own"
        ],
        index=(
            0
            if st.session_state.document_mode
            == "Built-in Documents"
            else 1
        )
    )

    # Store current mode
    st.session_state.document_mode = document_mode


    # ========================================================
    # BUILT-IN DOCUMENTS
    # ========================================================

    if document_mode == "Built-in Documents":

        if st.session_state.using_uploaded_documents:

            with st.spinner(
                "Restoring built-in documents..."
            ):

                rag.reset_to_builtin_documents()

            st.session_state.using_uploaded_documents = False
            st.session_state.uploaded_signature = None

            st.success(
                "Built-in documents restored."
            )

        else:

            st.info(
                "Using the example documents included "
                "with the application."
            )


    # ========================================================
    # UPLOAD YOUR OWN DOCUMENTS
    # ========================================================

    else:

        uploaded_files = st.file_uploader(
            "Upload PDF or TXT documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help=(
                "Upload one or more PDF or TXT documents. "
                "The system will extract, chunk, embed, "
                "and index them automatically."
            )
        )


        if uploaded_files:

            # ------------------------------------------------
            # CREATE FILE SIGNATURE
            # ------------------------------------------------

            signature_data = []

            for file in uploaded_files:

                file_bytes = file.getvalue()

                file_hash = hashlib.md5(
                    file_bytes
                ).hexdigest()

                signature_data.append(
                    (
                        file.name,
                        len(file_bytes),
                        file_hash
                    )
                )


            signature = hashlib.md5(
                repr(signature_data).encode("utf-8")
            ).hexdigest()


            # ------------------------------------------------
            # PROCESS ONLY IF FILES CHANGED
            # ------------------------------------------------

            if (
                st.session_state.uploaded_signature
                != signature
            ):

                with st.spinner(
                    "Extracting text, creating chunks, "
                    "and building the vector index..."
                ):

                    try:

                        rag.load_uploaded_documents(
                            uploaded_files
                        )

                        st.session_state.uploaded_signature = (
                            signature
                        )

                        st.session_state.using_uploaded_documents = (
                            True
                        )

                        st.success(
                            f"{len(uploaded_files)} document(s) "
                            "processed successfully."
                        )

                    except Exception as e:

                        st.error(
                            "Could not process the uploaded "
                            f"documents: {e}"
                        )

        else:

            st.info(
                "Upload one or more PDF or TXT files "
                "to create a custom knowledge base."
            )


    # ========================================================
    # SYSTEM INFORMATION
    # ========================================================

    st.divider()

    st.header("System Information")

    st.write(
        f"Documents: {len(rag.documents)}"
    )

    st.write(
        f"Chunks: {len(rag.chunks)}"
    )

    st.write("Embedding model:")

    st.code(
        "all-MiniLM-L6-v2"
    )

    st.write("Generator:")

    st.code(
        "FLAN-T5-base"
    )

    st.write("Vector database:")

    st.code(
        "FAISS"
    )

    st.divider()

    st.caption(
        "Answers are generated using retrieved "
        "document evidence."
    )


# ============================================================
# MAIN DOCUMENT STATUS
# ============================================================

st.subheader("Current Knowledge Base")


if st.session_state.using_uploaded_documents:

    st.markdown(
        '<div class="success-box">'
        '<b>Custom documents active</b><br>'
        'The system is answering questions using '
        'your uploaded documents.'
        '</div>',
        unsafe_allow_html=True
    )

else:

    st.markdown(
        '<div class="info-box">'
        '<b>Built-in documents active</b><br>'
        'You can ask questions about the example '
        'documents included with the application.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# WHAT CAN I ASK?
# ============================================================

st.subheader("What can I ask?")

st.write(
    "Ask questions that can be answered from the "
    "content of the selected documents."
)

example_col1, example_col2 = st.columns(2)


with example_col1:

    st.markdown(
        """
        **Examples**

        - What is the main purpose of this document?
        - What are the key requirements?
        - Who is eligible?
        - What are the important deadlines?
        """
    )


with example_col2:

    st.markdown(
        """
        **You can also ask**

        - What exceptions are mentioned?
        - What documents are required?
        - How many days are allowed?
        - What happens if a requirement is not met?
        """
    )


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("Ask a Question")

question = st.text_input(
    "Question",
    placeholder=(
        "e.g. What are the key requirements "
        "mentioned in this document?"
    ),
    label_visibility="collapsed"
)


# ============================================================
# GET ANSWER
# ============================================================

if st.button(
    "Get Answer",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif len(rag.documents) == 0:

        st.warning(
            "No documents are currently available. "
            "Please upload a PDF/TXT file or use the "
            "built-in documents."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                response = rag.answer(
                    question,
                    top_k=TOP_K,
                    score_threshold=SCORE_THRESHOLD
                )

            except Exception as e:

                st.error(
                    "An error occurred while generating "
                    f"the answer: {e}"
                )

                response = None


        # ====================================================
        # DISPLAY ANSWER
        # ====================================================

        if response is not None:

            st.subheader("Answer")

            st.markdown(
                '<div class="answer-box">'
                + response["answer"]
                + '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # DISPLAY EVIDENCE
            # =================================================

            st.subheader(
                "Retrieved Evidence"
            )


            if response["results"]:

                st.caption(
                    "These document chunks were retrieved "
                    "and provided to the language model."
                )


                for i, result in enumerate(
                    response["results"],
                    start=1
                ):

                    chunk = result["chunk"]

                    source_name = os.path.basename(
                        chunk.source
                    )

                    similarity = result["score"]


                    with st.expander(
                        f"Evidence {i} — "
                        f"{source_name} — "
                        f"Similarity: {similarity:.4f}"
                    ):

                        st.write(
                            f"**Document ID:** "
                            f"{chunk.doc_id}"
                        )

                        st.write(
                            f"**Chunk ID:** "
                            f"{chunk.chunk_id}"
                        )

                        st.write(
                            f"**Similarity Score:** "
                            f"{similarity:.4f}"
                        )

                        st.markdown(
                            "**Retrieved Text:**"
                        )

                        st.write(
                            chunk.text
                        )


            else:

                st.info(
                    "No sufficiently relevant evidence "
                    "was found in the selected documents."
                )


# ============================================================
# FAILURE CASE
# ============================================================

st.divider()

st.caption(
    "If the retrieved evidence does not contain "
    "the answer, the system responds: "
    '"I don\'t know from the provided documents."'
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Sentence Transformers, FAISS, "
    "FLAN-T5 and Streamlit."
)