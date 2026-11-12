import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configuration
sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (10, 6)

print("="*70)
print("  ANALYSE DES PRIX IMMOBILIERS - RÉGRESSION LINÉAIRE")
print("="*70)

# ==========================================
# ÉTAPE 1 : CHARGEMENT
# ==========================================

print("\n📂 ÉTAPE 1 : Chargement des données")
print("-"*70)

# Charger le dataset
df = pd.read_csv('data/Housing.csv')

print(f"✅ Données chargées : {df.shape[0]} observations, {df.shape[1]} variables")

print(f"\n📋 Variables disponibles :")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\n📊 Aperçu des 5 premières lignes :")
print(df.head())

print(f"\n📈 Types de données :")
print(df.dtypes)

print(f"\n📊 Statistiques descriptives :")
print(df.describe())

# Valeurs manquantes
print(f"\n🔍 Valeurs manquantes :")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✅ Aucune valeur manquante")
else:
    print(missing[missing > 0])

print("\n✅ ÉTAPE 1 TERMINÉE\n")

# ==========================================
# ÉTAPE 2 : EXPLORATION
# ==========================================

print("="*70)
print("📊 ÉTAPE 2 : Analyse Exploratoire")
print("-"*70)

# La colonne s'appelle 'prix'
price_col = 'prix'
print(f"✅ Colonne de prix identifiée : '{price_col}'")

# Distribution du prix
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df[price_col], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Prix (MAD)', fontsize=12)
plt.ylabel('Fréquence', fontsize=12)
plt.title('Distribution des Prix', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df[price_col])
plt.ylabel('Prix (MAD)', fontsize=12)
plt.title('Boxplot des Prix', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('resultats/01_distribution_prix.png', dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé : resultats/01_distribution_prix.png")
plt.close()

# Sélectionner uniquement colonnes numériques
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_cols]

print(f"\n📊 Variables numériques : {list(numeric_cols)}")

# Matrice de corrélation
plt.figure(figsize=(12, 10))
corr_matrix = df_numeric.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matrice de Corrélation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('resultats/02_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé : resultats/02_correlation_matrix.png")
plt.close()

print("\n📊 Corrélations avec le Prix :")
correlations = df_numeric.corr()[price_col].sort_values(ascending=False)
print(correlations)

print("\n✅ ÉTAPE 2 TERMINÉE\n")

# ==========================================
# ÉTAPE 3 : PRÉTRAITEMENT
# ==========================================

print("="*70)
print("🔧 ÉTAPE 3 : Prétraitement des Données")
print("-"*70)

# Encoder les variables catégorielles si nécessaire
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"\n📋 Variables catégorielles : {list(categorical_cols)}")

if len(categorical_cols) > 0:
    print("🔄 Encodage des variables catégorielles...")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"✅ Encodage terminé : {df_encoded.shape[1]} colonnes après encodage")
else:
    df_encoded = df.copy()
    print("✅ Aucune variable catégorielle à encoder")

# Séparation X et y
if price_col:
    y = df_encoded[price_col]
    X = df_encoded.drop(price_col, axis=1)
    
    print(f"\n✅ Variable cible (y) : {price_col}")
    print(f"✅ Variables explicatives (X) : {X.shape[1]} variables")
    print(f"   {list(X.columns[:5])}... (5 premières)")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n✅ Données d'entraînement : {X_train.shape[0]} observations")
    print(f"✅ Données de test : {X_test.shape[0]} observations")

print("\n✅ ÉTAPE 3 TERMINÉE\n")

# ==========================================
# ÉTAPE 4 : RÉGRESSION LINÉAIRE SIMPLE
# ==========================================

print("="*70)
print("📐 ÉTAPE 4 : Régression Linéaire SIMPLE")
print("-"*70)

if price_col and X.shape[1] > 0:
    # Choisir la variable la plus corrélée
    if price_col in df_numeric.columns:
        correlations = df_numeric.corr()[price_col].abs().sort_values(ascending=False)
        # Exclure la variable cible elle-même
        best_var = correlations.index[1]
        
        print(f"🎯 Variable la plus corrélée avec le prix : {best_var}")
        print(f"   Corrélation : {correlations[best_var]:.3f}")
        
        # Régression simple
        X_simple = df_numeric[[best_var]].values
        y_simple = df_numeric[price_col].values
        
        # Split
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_simple, y_simple, test_size=0.2, random_state=42
        )
        
        # Modèle
        model_simple = LinearRegression()
        model_simple.fit(X_train_s, y_train_s)
        
        # Prédictions
        y_pred_train_s = model_simple.predict(X_train_s)
        y_pred_test_s = model_simple.predict(X_test_s)
        
        # Métriques
        r2_train = r2_score(y_train_s, y_pred_train_s)
        r2_test = r2_score(y_test_s, y_pred_test_s)
        rmse_test = np.sqrt(mean_squared_error(y_test_s, y_pred_test_s))
        
        print(f"\n📊 RÉSULTATS Régression Simple :")
        print(f"   Coefficient (β₁) : {model_simple.coef_[0]:.2f}")
        print(f"   Intercept (β₀) : {model_simple.intercept_:.2f}")
        print(f"   R² Train : {r2_train:.3f}")
        print(f"   R² Test : {r2_test:.3f}")
        print(f"   RMSE Test : {rmse_test:.2f}")
        
        # Graphique
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test_s, y_test_s, alpha=0.5, label='Données réelles')
        plt.plot(X_test_s, y_pred_test_s, color='red', linewidth=2, label='Prédictions')
        plt.xlabel(best_var, fontsize=12)
        plt.ylabel(price_col, fontsize=12)
        plt.title(f'Régression Simple : {price_col} vs {best_var}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('resultats/03_regression_simple.png', dpi=300, bbox_inches='tight')
        print("✅ Graphique sauvegardé : resultats/03_regression_simple.png")
        plt.close()

print("\n✅ ÉTAPE 4 TERMINÉE\n")

# ==========================================
# ÉTAPE 5 : RÉGRESSION LINÉAIRE MULTIPLE
# ==========================================

print("="*70)
print("📐 ÉTAPE 5 : Régression Linéaire MULTIPLE")
print("-"*70)

if price_col and X_train.shape[0] > 0:
    # Modèle multiple
    model_multiple = LinearRegression()
    model_multiple.fit(X_train, y_train)
    
    # Prédictions
    y_pred_train = model_multiple.predict(X_train)
    y_pred_test = model_multiple.predict(X_test)
    
    # Métriques
    r2_train_m = r2_score(y_train, y_pred_train)
    r2_test_m = r2_score(y_test, y_pred_test)
    rmse_test_m = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test_m = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\n📊 RÉSULTATS Régression Multiple :")
    print(f"   R² Train : {r2_train_m:.3f}")
    print(f"   R² Test : {r2_test_m:.3f}")
    print(f"   RMSE Test : {rmse_test_m:.2f}")
    print(f"   MAE Test : {mae_test_m:.2f}")
    
    # Top 5 coefficients
    coef_df = pd.DataFrame({
        'Variable': X.columns,
        'Coefficient': model_multiple.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print(f"\n📊 Top 5 Variables les plus influentes :")
    print(coef_df.head())
    
    # Graphique prédictions vs réel
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Prédiction parfaite')
    plt.xlabel('Prix Réel', fontsize=12)
    plt.ylabel('Prix Prédit', fontsize=12)
    plt.title('Régression Multiple : Prédictions vs Réel', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('resultats/04_regression_multiple.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé : resultats/04_regression_multiple.png")
    plt.close()

print("\n✅ ÉTAPE 5 TERMINÉE\n")

print("="*70)
print("  🎉 ANALYSE COMPLÈTE TERMINÉE")
print("="*70)
print("\n📁 Résultats sauvegardés dans 'resultats/'")
print("📊 Prochaine étape : Validation des hypothèses du modèle")
