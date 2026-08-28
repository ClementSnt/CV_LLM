# CV Chatbot 🤖


🔗 **[Accéder au dashboard interactif](https://geronimo-llm-chatbot.streamlit.app/)**

Projet personnel visant à comprendre un peu mieux le fonctionnement des modèles LLM en créant un chatbot capable de répondre à des questions sur mon CV, mes expériences, compétences et projets.

Le principe est assez simple : Sentence Transformers permet d'identifier les informations de mon CV les plus proches de la question posée, puis Flan-T5 Small utilise ce contexte pour générer une réponse en anglais.

Exemples de questions :
What are your main data science skills?
What do you do at Disney?
What projects have you worked on?
What are your hobbies?

Les réponses restent assez décevantes, mais je suis limité en puissance de calcul puisque j'utilise Streamlit Cloud. Le modèle utilisé est donc volontairement léger (et un peu bête du coup), d'autant que la qualité des réponses dépend également fortement des informations présentes dans la base documentaire.
