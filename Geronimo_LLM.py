import os
import streamlit as st
import torch

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Clément's CV Chatbot",
    page_icon="🤖"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-small"


# --------------------------------------------------
# Chargement des données
# --------------------------------------------------

@st.cache_data
def load_documents():

    documents = []

    for filename in os.listdir("data"):

        if filename.endswith(".txt"):

            path = os.path.join("data", filename)

            with open(path, "r", encoding="utf-8") as file:
                text = file.read().strip()

            documents.append(text)

    return documents


# --------------------------------------------------
# Chargement du modèle d'embeddings
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(EMBEDDING_MODEL)


# --------------------------------------------------
# Chargement de FLAN-T5
# --------------------------------------------------

@st.cache_resource
def load_generator():

    tokenizer = AutoTokenizer.from_pretrained(
        GENERATION_MODEL
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        GENERATION_MODEL
    )

    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return generator


# --------------------------------------------------
# Création des embeddings
# --------------------------------------------------

@st.cache_resource
def create_embeddings(_embedding_model, documents):

    return _embedding_model.encode(
        list(documents),
        convert_to_tensor=True,
        normalize_embeddings=True
    )


# --------------------------------------------------
# Recherche des informations pertinentes
# --------------------------------------------------

def retrieve_context(
    question,
    documents,
    embedding_model,
    document_embeddings,
    n_results=2
):

    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    scores = util.cos_sim(
        question_embedding,
        document_embeddings
    )[0]

    top_results = torch.topk(
        scores,
        k=min(n_results, len(documents))
    )

    selected_documents = [
        documents[index]
        for index in top_results.indices.tolist()
    ]

    return "\n\n".join(selected_documents)


# --------------------------------------------------
# Réponse du chatbot
# --------------------------------------------------

def ask_cv_bot(
    question,
    documents,
    embedding_model,
    document_embeddings,
    generator
):

    context = retrieve_context(
        question,
        documents,
        embedding_model,
        document_embeddings
    )

    prompt = f"""
Answer the recruiter's question using only the information in the context.

Speak as Clément using "I".
Answer in English.
Be concise and natural.
Do not invent information.
If the answer is not in the context, say that you do not have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False
    )

    return result[0]["generated_text"].strip()


# --------------------------------------------------
# Initialisation
# --------------------------------------------------

documents = load_documents()

embedding_model = load_embedding_model()

document_embeddings = create_embeddings(
    embedding_model,
    tuple(documents)
)

generator = load_generator()


# --------------------------------------------------
# Interface Streamlit
# --------------------------------------------------

st.title("Clément's CV Chatbot 🤖")

st.write(
    "Ask me questions about my experience, skills, projects or education."
)

question = st.text_input(
    "Your question:",
    placeholder="What experience do you have in forecasting?"
)


if question:

    with st.spinner("Thinking..."):

        answer = ask_cv_bot(
            question,
            documents,
            embedding_model,
            document_embeddings,
            generator
        )

    st.markdown(f"**Answer:** {answer}")
