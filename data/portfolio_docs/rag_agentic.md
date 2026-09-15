# RAG & agentic AI with LangChain

Quatrième projet du parcours AI/ML — refaire, avec LangChain, l'équivalent du co-pilote agentique construit pendant le stage OCP Jorf Lasfar, pour comparer une implémentation custom à un framework standard du marché.

## Objectif

- Comparer un pipeline RAG hybride custom (ChromaDB + BM25, projet OCP) à l'implémentation équivalente avec LangChain
- Construire un agent capable de choisir lui-même entre plusieurs outils selon la question posée (recherche documentaire vs génération de checklist)

## Architecture

**Partie 1 — Retrieval (`src/retrieval.py`, testable sans LLM)**
- Chargement et chunking de documents (`RecursiveCharacterTextSplitter`)
- Retrieval hybride : `BM25Retriever` (mots-clés) + `FAISS` avec embeddings TF-IDF maison (similarité), combinés via `EnsembleRetriever`
- Équivalent du hybride ChromaDB+BM25 construit à l'OCP, mais avec les abstractions standards de LangChain

**Partie 2 — Agent (`src/agent.py`, nécessite Ollama)**
- Deux outils : `search_hse_documents` (RAG de la partie 1) et `generate_incident_checklist` (génération structurée)
- `create_agent` de LangChain orchestre la décision : l'agent choisit lui-même quel outil appeler selon la requête
- Modèle : Qwen2.5 3B via Ollama (même modèle que le co-pilote OCP)

## Comparaison avec l'implémentation OCP

| | Custom (OCP) | LangChain (ce projet) |
|---|---|---|
| Retrieval | ChromaDB + BM25 fait main | `EnsembleRetriever` (BM25Retriever + FAISS) |
| Orchestration | Logique custom en FastAPI | `create_agent` avec function calling |
| Modèle | Qwen2.5 3B (Ollama) | Qwen2.5 3B (Ollama) — identique |
| Avantage custom | Contrôle total, pas de dépendance framework | — |
| Avantage LangChain | Moins de code, abstractions testées, plus rapide à itérer | — |

## Lancer le projet

```bash
pip install -r requirements.txt

# Partie 1 (fonctionne partout, aucune dépendance externe)
cd src
python retrieval.py

# Partie 2 (nécessite Ollama)
# 1. Installer Ollama : https://ollama.com
# 2. ollama pull qwen2.5:3b
# 3. python agent.py
```

## Prochaines améliorations possibles

- Remplacer les embeddings TF-IDF par des embeddings neuronaux locaux (`nomic-embed-text` via Ollama)
- Ajouter un outil de web search pour les questions hors base documentaire
- Comparer les temps de réponse custom vs LangChain sur un même corpus de test
