# Modélisation Prédictive de Prix Immobiliers — Pipeline de Régression

Projet complet de Machine Learning supervisé visant à prédire la valeur de biens immobiliers à partir de leurs caractéristiques structurelles et géographiques.

## Architecture du Projet

* `data/` : Jeux de données bruts et prétraités.
* `resultats/` : Courbes d'évaluation, métriques et prédictions exportées.
* `download_data.py` : Ingestion automatisée du dataset.
* `main.py` : Pipeline ML complet (nettoyage, entraînement, validation et export).
* `test_install.py` : Vérification de l'environnement et des dépendances.
* `requirements.txt` : Liste des bibliothèques Python requises.

## Méthodologie & Stack Technique

* **Langage & Bibliothèques :** Python, Pandas, NumPy, Scikit-learn, Matplotlib.
* **Étapes du pipeline :**
  * **Prétraitement :** Gestion des valeurs manquantes, encodage des variables catégorielles et normalisation des features.
  * **Modélisation :** Entraînement et comparaison d'algorithmes de régression.
  * **Évaluation :** Analyse des résidus et validation des performances (RMSE, MAE, $R^2$).

## Installation et Exécution

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Télécharger le jeu de données
python download_data.py

# 3. Lancer l'entraînement et générer les résultats
python main.py
