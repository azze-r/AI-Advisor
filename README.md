<img width="928" alt="image" src="https://github.com/InsightEdge01/RAGGemmaModel/assets/131486782/0645d193-b59a-4809-8082-1af3aad80aa3">

Dans Terminal.
Creation de l'environnement virtuel
python3 -m venv .venv  
Activation de l'environnement virtuel
.\.venv\Scripts\activate

pip install -r requirements.txt
ollama pull  gemma:7b
ollama pull nomic-embed-text
chainlit run app.py