# Titanic survival classifier

Premier projet du parcours AI/ML — prédire si un passager du Titanic a survécu, à partir de ses caractéristiques (classe, sexe, âge, etc.).

## Objectif

Comprendre et appliquer le workflow de base du Machine Learning supervisé sur des données réelles : nettoyage, encodage de variables catégorielles, split train/test, entraînement, évaluation.

## Données

[Titanic dataset](https://www.kaggle.com/c/titanic) — 891 passagers, avec des valeurs manquantes réelles (Age, Cabin, Embarked) qu'il a fallu traiter.

## Approche

- Remplissage des valeurs manquantes (médiane pour l'âge, valeur la plus fréquente pour le port d'embarquement)
- Encodage des variables catégorielles (Sex, Embarked) en variables numériques
- Modèle : régression logistique (`scikit-learn`)
- Split 80/20 avec stratification sur la variable cible

## Résultats

| Métrique | Score |
|----------|-------|
| Accuracy | 0.804 |
| Precision | 0.793 |
| Recall | 0.667 |

Les facteurs les plus déterminants pour la survie : le sexe du passager (le plus influent), suivi de la classe (Pclass).

## Lancer le projet

```bash
pip install -r requirements.txt
cd src
python train.py
```

## Prochaines améliorations possibles

- Tester d'autres modèles (Random Forest, XGBoost) et comparer
- Feature engineering (extraire le titre depuis le nom, taille de famille)
- Cross-validation au lieu d'un simple split
