"""
Capstone (partie 3/3) — Agent complet : matching + RAG + génération de lettre.

A LANCER CHEZ TOI (nécessite Ollama) :
    ollama pull qwen2.5:3b
    python agent.py

L'agent orchestre 3 outils : il analyse l'offre, retrouve les projets
pertinents dans ton portfolio, PUIS rédige une lettre qui cite des
détails précis (pas des généralités) -- c'est le tool-calling qui
fait la différence avec un simple prompt ChatGPT.
"""

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from matching import load_profile, analyze_match
from portfolio_retrieval import search_portfolio

profile = load_profile()


@tool
def analyze_job_match(job_description: str) -> str:
    """Analyse une offre de PFE/stage et calcule le score d'adéquation
    avec le profil de Rajae, en listant les compétences qui correspondent
    et les lacunes potentielles."""
    result = analyze_match(job_description, profile)
    return (
        f"Score de matching : {result['match_score']}%\n"
        f"Compétences qui correspondent : {', '.join(result['matched_skills'])}\n"
        f"Lacunes potentielles : {', '.join(result['potential_gaps'][:5])}"
    )


@tool
def search_my_portfolio(query: str) -> str:
    """Recherche dans le portfolio de projets de Rajae (READMEs des 5 projets
    AI/ML) pour trouver des détails techniques précis à citer dans une lettre
    de motivation."""
    results = search_portfolio(query, top_k=3)
    return "\n\n".join(f"[{r['source']}] {r['content']}" for r in results)


@tool
def get_experience_summary() -> str:
    """Retourne un résumé de l'expérience professionnelle de Rajae
    (stages OCP et SONASID)."""
    return "\n\n".join(
        f"{e['role']} chez {e['company']} ({e['dates']}) : {e['summary']}"
        for e in profile["experience"]
    )


llm = ChatOllama(model="qwen2.5:3b", temperature=0.3)

agent = create_agent(
    model=llm,
    tools=[analyze_job_match, search_my_portfolio, get_experience_summary],
    system_prompt=(
        "Tu es l'assistant de recherche de PFE de Rajae Elkamili, étudiante "
        "ingénieure en dernière année spécialisée en AI/ML. "
        "Quand on te donne une offre de stage/PFE, utilise analyze_job_match "
        "pour évaluer l'adéquation, search_my_portfolio pour trouver des projets "
        "pertinents à mentionner, et get_experience_summary si besoin. "
        "Rédige ensuite une lettre de motivation concise (200-250 mots), "
        "professionnelle mais pas générique -- cite des détails techniques "
        "précis tirés du portfolio plutôt que des formules toutes faites. "
        "Réponds en français."
    ),
)

if __name__ == "__main__":
    example_job = """
    Nous recherchons un stagiaire PFE en Machine Learning et Deep Learning pour
    rejoindre notre équipe Data Science. Vous travaillerez sur des projets de
    NLP et de systèmes RAG (LangChain), avec déploiement en production via
    Docker et FastAPI. Connaissance de MLflow appréciée.
    """

    print("=== Analyse + génération de lettre ===\n")
    response = agent.invoke({
        "messages": [{"role": "user", "content": f"Voici une offre de PFE :\n{example_job}\n\nAnalyse-la et rédige-moi une lettre de motivation adaptée."}]
    })
    print(response["messages"][-1].content)
