# CV Chatbot 🤖


🔗 **[Accéder au dashboard interactif](https://geronimo-llm-chatbot.streamlit.app/)**

## Description
Ce projet vise à créer un chatbot capable de répondre à des questions sur mon CV, mes compétences, expériences, projets, hobbies, etc

Le chatbot utilise :
- ChromaDB pour stocker et rechercher les informations pertinentes contenues dans des documents segmentés
- Sentence-Transformers pour générer des embeddings et effectuer des recherches sémantiques
- Hugging Face Transformers (Flan-T5 Large) pour générer des réponses en langage naturel à partir du contexte récupéré
- Streamlit pour l’interface utilisateur web

## Limitations !!!!!
- Flan-T5 Large nécessite beaucoup de mémoire GPU et plante réguliérement sur Streamlit, je n'utilise pas d'autre plan payant pouvant compenser
- Les modèles plus légers (T5-Base, T5-Small) donnent des réponses quasiment systématiquement à côté de la plaque, cela peut aussi arriver avec T5-Large mais moins souvent. J'aurais aimé tester avec des modèles plus large et notament MistralAI mais beaucoup trop lourd pour mon setup
- Le chatbot ne peut répondre qu’aux questions dont les informations sont présentes dans les documents 


## Fonctionnement
1. L’utilisateur pose une question en anglais
2. Le chatbot recherche les documents les plus pertinents dans ChromaDB  
3. Flan-T5 génère une réponse

Exemples de questions :
- What are your main data science skills?  
- Can you explain your tasks at Disney?  
- What projects have you done?  
- What are your hobbies?




