"""
Téléchargement et adaptation du dataset au contexte marocain
"""
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

print("📥 Téléchargement du dataset...")

# Télécharger
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# ADAPTER AU CONTEXTE MAROCAIN
print("🇲🇦 Adaptation au contexte marocain...")

# Renommer les colonnes en français/contexte marocain
df.columns = ['revenu_median', 'age_bien', 'nb_pieces', 'nb_chambres',
              'population_quartier', 'nb_occupants', 'latitude', 'longitude', 'prix']

# Convertir les prix en MAD (multiplier par 1,000,000 pour avoir des prix réalistes)
# Prix original en centaines de milliers de $ → Prix en MAD
df['prix'] = (df['prix'] * 100000 * 10).round(0)  # 1$ = 10 MAD

# Convertir revenu en MAD/mois
df['revenu_median'] = (df['revenu_median'] * 10000 * 10 / 12).round(0)

# Sauvegarder
df.to_csv('data/housing.csv', index=False)

print(f"✅ Dataset adapté : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"✅ Fichier sauvegardé : data/housing.csv")
print("\n📊 Aperçu des données marocaines :")
print(df.head())
print("\n💰 Statistiques des prix (MAD) :")
print(df['prix'].describe())