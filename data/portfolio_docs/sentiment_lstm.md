# Sentiment LSTM classifier

Troisième projet du parcours AI/ML — analyse de sentiment sur des critiques de films IMDB avec un LSTM bidirectionnel construit en PyTorch.

## Objectif

Passer des images (projet 2) au texte : comprendre la tokenisation, les embeddings, et le traitement séquentiel avec un LSTM. Ce projet fait le pont naturel vers les LLMs.

## Données

[IMDB Dataset of 50K Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) (pré-traité), sous-échantillonné à 8000 critiques pour un entraînement rapide, équilibré 50/50 positif/négatif.

## Architecture

- Embedding appris (128 dimensions) sur un vocabulaire de 8000 mots
- LSTM bidirectionnel (128 unités, lit la séquence dans les deux sens)
- Dropout (0.3) + couche dense finale pour la classification binaire

## Méthodologie (et l'itération qui compte)

Une première version (5 epochs, hidden_dim=64) donnait 57% d'accuracy — sous-apprentissage. Augmenter la capacité et les epochs a fait monter l'accuracy à 76.6%, mais avec une train_loss tombant à 0.001 — signe clair d'overfitting.

**Solution appliquée : early stopping avec un set de validation séparé du test.** Le modèle est entraîné jusqu'à ce que la validation loss cesse de s'améliorer (patience = 3 epochs), puis on restaure les poids du meilleur epoch plutôt que ceux du dernier.

## Résultats

| Version | Test accuracy | Note |
|---------|---------------|------|
| Baseline (5 epochs, petit modèle) | 0.570 | Sous-apprentissage |
| Modèle poussé (20 epochs, bidirectional) | 0.766 | Overfitting (train_loss → 0.001) |
| **Avec early stopping** | **0.738** | Score honnête, non gonflé |

## Lancer le projet

```bash
pip install torch pandas scikit-learn
cd src
python train.py
```

## Prochaines améliorations possibles

- Utiliser des embeddings pré-entraînés (GloVe/Word2Vec) au lieu d'un embedding appris de zéro
- Passer à un Transformer léger (DistilBERT) pour comparer les performances
- Entraîner sur le dataset complet (50000 critiques) plutôt qu'un sous-échantillon
