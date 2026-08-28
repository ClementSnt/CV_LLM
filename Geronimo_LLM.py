import os
import streamlit as st
import torch

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Clément's CV Chatbot",
    page_icon="🤖"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Plus léger que flan-t5-large
GENERATION_MODEL = "google/flan-t5-base"


# ---------------------------------------------------------
# LOAD DOCUMENTS
# ---------------------------------------------------------

@st.cache_data
def load_documents():
    documents = []

    for filename in os.listdir("data"):
        if filename.endswith(".txt"):
            path = os.path.join("data", filename)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()

                documents.append({
                    "source": filename,
                    "text": text
                })

    return documents


# ---------------------------------------------------------
# SPLIT DOCUMENTS INTO SMALLER CHUNKS
# ---------------------------------------------------------

@st.cache_data
def create_chunks(documents, chunk_size=700):
    chunks = []

    for document in documents:
        text = document["text"]

        # Découpe simple par paragraphes
        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]

        current_chunk = ""

        for paragraph in paragraphs:

            if len(current_chunk) + len(paragraph) <= chunk_size:
                current_chunk += "\n" + paragraph

            else:
                if current_chunk:
                    chunks.append({
                        "source": document["source"],
                        "text": current_chunk.strip()
                    })

                current_chunk = paragraph

        if current_chunk:
            chunks.append({
                "source": document["source"],
                "text": current_chunk.strip()
            })

    return chunks


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------
# CREATE EMBEDDINGS
# ---------------------------------------------------------

@st.cache_resource
def create_embeddings(_embedding_model, texts):
    return _embedding_model.encode(
        list(texts),
        convert_to_tensor=True,
        normalize_embeddings=True
    )


# ---------------------------------------------------------
# LOAD GENERATION MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_generator():

    tokenizer = AutoTokenizer.from_pretrained(
        GENERATION_MODEL
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        GENERATION_MODEL
    )

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer
    )


# ---------------------------------------------------------
# RETRIEVE RELEVANT INFORMATION
# ---------------------------------------------------------

def retrieve_context(
    query,
    embedding_model,
    chunk_embeddings,
    chunks,
    n_results=3
):

    query_embedding = embedding_model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    scores = util.cos_sim(
        query_embedding,
        chunk_embeddings
    )[0]

    top_results = torch.topk(
        scores,
        k=min(n_results, len(chunks))
    )

    selected_chunks = []

    for score, index in zip(
        top_results.values,
        top_results.indices
    ):

        selected_chunks.append({
            "score": float(score),
            "source": chunks[index]["source"],
            "text": chunks[index]["text"]
        })

    return selected_chunks


# ---------------------------------------------------------
# GENERATE ANSWER
# ---------------------------------------------------------

def ask_cv_bot(
    query,
    embedding_model,
    chunk_embeddings,
    chunks,
    generator
):

    retrieved = retrieve_context(
        query,
        embedding_model,
        chunk_embeddings,
        chunks,
        n_results=3
    )

    # Si aucun résultat n'est vraiment proche,
    # éviter que le bot invente
    if retrieved[0]["score"] < 0.25:
        return (
            "I don't have enough information in my CV "
            "to answer this question accurately."
        )

    context = "\n\n".join(
        item["text"]
        for item in retrieved
    )

    prompt = f"""
You are Clément's CV assistant.

Your job is to answer questions about Clément's professional experience,
education, skills and personal projects.

Rules:
- Answer in English.
- Speak in the first person ("I").
- Answer naturally, as if Clément was answering a recruiter.
- Keep the answer concise: usually 2 to 4 sentences.
- Use only information provided in the context.
- Do not invent experience, skills or achievements.
- If the context does not contain enough information, say so.
- Prefer concrete facts over generic statements.
- When relevant, connect related information from several parts of the context.

Context:
{context}

Recruiter's question:
{query}

Answer:
"""

    output = generator(
        prompt,
        max_new_tokens=130,
        do_sample=False,
        num_beams=2,
        repetition_penalty=1.15
    )

    return output[0]["generated_text"].strip()


# ---------------------------------------------------------
# INITIALISATION
# ---------------------------------------------------------

documents = load_documents()

chunks = create_chunks(documents)

embedding_model = load_embedding_model()

chunk_texts = tuple(
    chunk["text"]
    for chunk in chunks
)

chunk_embeddings = create_embeddings(
    embedding_model,
    chunk_texts
)

generator = load_generator()


# ---------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------

st.title("Clément's CV Chatbot 🤖")

st.write(
    """
Ask me questions about my experience, skills,
education or projects.
"""
)

user_question = st.text_input(
    "Your question:",
    placeholder="What experience do you have in forecasting?"
)

if user_question:

    with st.spinner("Thinking..."):

        answer = ask_cv_bot(
            user_question,
            embedding_model,
            chunk_embeddings,
            chunks,
            generator
        )

    st.markdown(f"**Answer:** {answer}")
