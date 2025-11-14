# 1️⃣ Imports
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 2️⃣ Configuration Chroma
chroma_client = chromadb.Client()

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
    print("✅ Collection 'cv_llm' created and filled!")

# 3️⃣ Modèle Flan-T5 Large en 8-bit
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    load_in_8bit=True,      # réduit la mémoire GPU
    device_map="auto"       # utilise GPU si dispo
)

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer
)

# 4️⃣ Fonction de question
def ask_cv_bot_en(query, n_docs=2):
    # Recherche documents
    results = collection.query(query_texts=[query], n_results=n_docs)
    
    if len(results["documents"]) == 0 or len(results["documents"][0]) == 0:
        return "Sorry, I don't have information to answer that."
    
    context = " ".join(results["documents"][0][:n_docs])

    prompt = f"""
Answer the question below using only the context. 
Do not invent anything. Be concise, factual, and use full sentences.

Context:
{context}

Question: {query}
Answer:
"""
    answer = generator(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.8
    )[0]["generated_text"]

    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    return answer

# 5️⃣ Interface Streamlit
st.set_page_config(page_title="CV Chatbot", page_icon="🤖")
st.title("Clément's CV Chatbot 🤖")
st.write("Ask me questions about my experience, skills, projects or hobbies. Responses might take a few seconds.")

user_question = st.text_input("Your question in English:")

if user_question:
    answer = ask_cv_bot_en(user_question, n_docs=2)
    st.markdown(f"**Answer:** {answer}")



