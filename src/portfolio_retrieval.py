"""
Capstone (partie 2/3) — RAG sur ton propre portfolio.

Objectif : permettre à l'agent de retrouver les détails techniques
précis d'un de tes projets pour les citer dans une lettre de motivation
-- plutôt que de réciter une description générique.

Réutilise le pattern hybride BM25+FAISS du projet 4 (RAG & Agentic AI).
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer

loader = DirectoryLoader("../data/portfolio_docs", glob="*.md", loader_cls=TextLoader)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
chunks = splitter.split_documents(documents)

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3


class TfidfEmbeddings(Embeddings):
    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer().fit(texts)

    def embed_documents(self, texts):
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text):
        return self.vectorizer.transform([text]).toarray()[0].tolist()


all_texts = [c.page_content for c in chunks]
embeddings = TfidfEmbeddings(all_texts)
vectorstore = FAISS.from_documents(chunks, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

portfolio_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)


def search_portfolio(query, top_k=3):
    """Retrouve les passages les plus pertinents du portfolio pour une requête donnée."""
    results = portfolio_retriever.invoke(query)
    return [
        {"source": r.metadata["source"].split("/")[-1], "content": r.page_content}
        for r in results[:top_k]
    ]


if __name__ == "__main__":
    queries = [
        "expérience avec le déploiement de modèles et MLOps",
        "systèmes RAG et agents avec function calling",
        "gestion de l'overfitting dans un modèle",
    ]
    for q in queries:
        print(f"\n{'='*60}\nRequête : {q}\n{'='*60}")
        for r in search_portfolio(q, top_k=2):
            print(f"[{r['source']}] {r['content'][:150]}...")
