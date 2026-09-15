# PFE Search Assistant

un assistant agentic qui m'aide dans ma propre recherche de PFE : analyse d'offres, matching avec mon profil, et génération de lettres de motivation qui citent mes vrais projets plutôt que des généralités.

## Architecture

```
data/profile.json          — mon profil structuré (skills, expérience, projets)
data/portfolio_docs/       — READMEs de mes 5 projets AI/ML (base RAG)

src/matching.py             — score d'adéquation offre <-> profil (testable, sans LLM)
src/portfolio_retrieval.py  — RAG hybride (BM25+FAISS) sur mon portfolio
src/agent.py                — agent complet : matching + RAG + génération (nécessite Ollama)
src/api.py                  — API FastAPI pour la partie matching
```

## Ce que fait chaque partie

**Matching (`matching.py`)** — Compare une offre de stage à mon profil sur deux axes : correspondance exacte de compétences nommées, et similarité TF-IDF globale (contexte). Retourne un score composite + les compétences qui matchent + les lacunes potentielles réelles (testé : détecte correctement "Kubernetes" et "Computer Vision" comme lacunes sur un exemple).

**RAG portfolio (`portfolio_retrieval.py`)** — Réutilise le pattern hybride BM25+FAISS du projet 4, appliqué à mes propres READMEs de projets, pour retrouver le bon projet à citer selon le sujet de l'offre.

**Agent (`agent.py`)** — Orchestre 3 outils (`analyze_job_match`, `search_my_portfolio`, `get_experience_summary`) via LangChain + Qwen2.5 (Ollama) pour produire une lettre de motivation qui cite des détails techniques précis plutôt que des formules génériques.

**API (`api.py`)** — Expose la partie matching (testable sans LLM) via FastAPI.

## Exemple de résultat (matching, testé)

Offre : *"Stage PFE en Machine Learning, PyTorch, Docker, deployment MLOps, RAG systems"*

```json
{
  "match_score": 36.1,
  "matched_skills": ["Docker", "Machine Learning", "PyTorch", "RAG"],
  "potential_gaps": ["deployment", "mlops", "systems"]
}
```
(Score plus bas ici car l'offre est très courte — le score TF-IDF pondère la richesse du contexte, pas seulement les mots-clés exacts.)

## Lancer le projet

```bash
pip install -r requirements.txt
cd src

# Matching (fonctionne partout, testé)
python matching.py

# RAG sur le portfolio (fonctionne partout, testé)
python portfolio_retrieval.py

# API matching (fonctionne partout)
uvicorn api:app --reload --port 8000

# Agent complet avec génération de lettre (nécessite Ollama)
# 1. ollama pull qwen2.5:3b
# 2. python agent.py
```

## Prochaines améliorations possibles

- Interface web simple (Streamlit) pour coller une offre et obtenir score + lettre en un clic
- Historique des candidatures avec statut (envoyée, réponse reçue, entretien)
- Scraping automatique d'offres depuis LinkedIn/Indeed (nécessite gestion des CGU de ces plateformes)
