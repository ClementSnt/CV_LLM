# Installations
!pip install -q sentence-transformers chromadb transformers accelerate

# Imports
import os
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configuration Chroma
chroma_client = chromadb.Client()

# Récupérer la collection
try:
    collection = chroma_client.get_collection("cv_llm")
except:
    collection = chroma_client.create_collection(name="cv_llm")

    # Charger les fichiers textes
    documents = []
    sources = []
    for file in os.listdir("data"):
        path = f"data/{file}"
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(content)
            sources.append(file)

    # Générer embeddings
    model_embed = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model_embed.encode(documents).tolist()

    # Ajouter à la collection
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=[{"source": s} for s in sources]
    )


# Modèle Flan-T5 Large
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer
)

# Fonction pour poser une question
def ask_cv_bot_en(query, n_docs=2):
    # Recherche des documents les plus pertinents
    results = collection.query(query_texts=[query], n_results=n_docs)
    context = " ".join(results["documents"][0][:n_docs])

    # Prompt optimisé pour un ton chaleureux et concis
    prompt = f"""
You are Clément, a business analyst and data scientist. 
Answer the question below **in English only**, using "I" statements. 
Be concise, friendly, and natural. 
Use the context to create **complete sentences**, do not list bullet points, and **do not add any information that is not present in the context**. 
Stick strictly to the facts provided.



Context:
{context}

Question: {query}
Answer:
"""

    answer = generator(
        prompt,
        max_new_tokens=120,   # limite la longueur
        do_sample=True,       # active la génération aléatoire pour plus de naturel
        temperature=0.8,      # variation du style
        top_p=0.8             # nucleus sampling
    )[0]["generated_text"]

    # Nettoyage : ne garder que la partie après "Answer:"
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    return answer


# Interface du chatbot Geronimo
import streamlit as st

st.set_page_config(page_title="CV Chatbot", page_icon="🤖")

st.title("Clément's CV Chatbot 🤖")
st.write("Ask me questions about my experience, skills, projects, or hobbies.")

# Champ pour saisir la question
user_question = st.text_input("Your question:", "")

if user_question:
    # Appel de la fonction que tu as déjà définie
    answer = ask_cv_bot_en(user_question, n_docs=2)
    st.markdown(f"**Answer:** {answer}")


