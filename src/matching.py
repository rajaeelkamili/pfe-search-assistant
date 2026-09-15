"""
Capstone (partie 1/3) — Matching profil <-> offre de stage.

Objectif : analyser une offre de PFE et calculer un score d'adéquation
avec le profil, en identifiant les compétences qui matchent et celles
qui manquent -- utile pour prioriser les candidatures ET pour savoir
quoi mettre en avant dans la lettre de motivation.

Cette partie ne nécessite AUCUN LLM -- c'est du matching lexical/TF-IDF,
transparent et explicable (contrairement à "l'IA a dit 80%" sans détail).
"""

import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_profile(path="../data/profile.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_skills(profile):
    """Rassemble toutes les compétences du profil en une seule liste."""
    all_skills = []
    for category in profile["skills"].values():
        all_skills.extend(category)
    return all_skills


def extract_keywords(text):
    """Extraction simple de mots-clés techniques -- tout mot/expression
    de 2+ caractères, en minuscule, sans ponctuation. Gère les accents français."""
    text = text.lower()
    words = re.findall(r"[a-zà-ÿ][a-zà-ÿ0-9\+\#\.]{2,}", text)
    return set(words)


def analyze_match(job_description, profile):
    """
    Calcule un score d'adéquation entre une offre et le profil.
    Combine deux signaux :
    1. Correspondance exacte de mots-clés (compétences techniques nommées)
    2. Similarité TF-IDF globale (capture le contexte, pas juste les mots exacts)
    """
    skills = flatten_skills(profile)
    job_keywords = extract_keywords(job_description)

    # 1. Correspondance exacte : quelles compétences du profil apparaissent dans l'offre ?
    matched_skills = []
    missing_relevant_skills = []
    for skill in skills:
        skill_lower = skill.lower()
        # gère les compétences multi-mots (ex: "Machine Learning")
        if skill_lower in job_description.lower():
            matched_skills.append(skill)

    # 2. Mots-clés de l'offre qui ne sont couverts par AUCUNE compétence du profil
    # (utile pour repérer les vraies lacunes, pas du bruit)
    profile_text = " ".join(skills).lower()
    french_stopwords = {"stage", "pfe", "entreprise", "avec", "pour", "dans", "notre",
                          "vous", "nous", "sera", "une", "des", "les", "sur", "aux",
                          "amené", "être", "atout", "appréciée", "connaissance",
                          "distribués", "systèmes", "équipe", "travaillerez", "recherchons"}
    tech_terms_in_job = [
        w for w in job_keywords
        if len(w) > 3 and w not in profile_text
        and w not in french_stopwords
    ]

    # 3. Similarité TF-IDF globale entre le profil (expérience+projets) et l'offre
    profile_full_text = " ".join([
        profile["education"],
        " ".join(e["summary"] for e in profile["experience"]),
        " ".join(profile["projects"]),
        " ".join(skills),
    ])
    vectorizer = TfidfVectorizer(stop_words=None)
    tfidf_matrix = vectorizer.fit_transform([profile_full_text, job_description])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Score composite : 60% correspondance exacte de compétences, 40% similarité globale
    keyword_score = min(len(matched_skills) / 8, 1.0)  # plafonné à 8 skills pour éviter un score gonflé
    composite_score = round((0.6 * keyword_score + 0.4 * similarity) * 100, 1)

    return {
        "match_score": composite_score,
        "matched_skills": sorted(set(matched_skills)),
        "potential_gaps": sorted(set(tech_terms_in_job))[:10],  # top 10 termes non couverts
        "tfidf_similarity": round(similarity * 100, 1),
    }


if __name__ == "__main__":
    profile = load_profile()

    # Exemple d'offre de PFE (à remplacer par une vraie offre copiée-collée)
    example_job = """
    Nous recherchons un stagiaire PFE en Machine Learning et Deep Learning pour
    rejoindre notre équipe Data Science. Vous travaillerez sur des projets de
    Computer Vision et NLP, en utilisant PyTorch et des frameworks de RAG
    (LangChain). Une expérience avec Docker et FastAPI est un plus. Connaissance
    de MLflow ou d'outils de tracking d'expériences appréciée. Vous serez amené
    à déployer des modèles en production et à collaborer avec l'équipe MLOps.
    Kubernetes et connaissance de systèmes distribués sont un atout.
    """

    result = analyze_match(example_job, profile)

    print(f"Score de matching : {result['match_score']}%\n")
    print(f"Compétences qui matchent ({len(result['matched_skills'])}) :")
    for skill in result["matched_skills"]:
        print(f"  ✓ {skill}")
    print(f"\nTermes potentiellement manquants dans le profil :")
    for gap in result["potential_gaps"]:
        print(f"  ? {gap}")
