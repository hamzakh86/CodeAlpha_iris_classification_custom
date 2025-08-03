#!/usr/bin/env python3
"""
Classification des fleurs d'Iris - Version Personnalisée
Auteur: Hamza Khaled
Date: 2025

Ce script implémente une solution complète de classification des fleurs d'Iris
avec des améliorations par rapport au projet de référence.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

class IrisClassifier:
    """Classe pour la classification des fleurs d'Iris avec des fonctionnalités avancées."""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
        
    def load_data(self):
        """Charger et préparer les données Iris."""
        print("📊 Chargement des données...")
        # Utiliser le dataset Iris de seaborn pour éviter les problèmes de chemin
        self.df = sns.load_dataset('iris')
        print(f"✅ Données chargées: {self.df.shape[0]} échantillons, {self.df.shape[1]} caractéristiques")
        return self.df
    
    def explore_data(self):
        """Explorer et visualiser les données."""
        print("\n🔍 Exploration des données...")
        print(f"Forme du dataset: {self.df.shape}")
        print(f"Colonnes: {list(self.df.columns)}")
        print(f"Types de données:\n{self.df.dtypes}")
        print(f"Valeurs manquantes:\n{self.df.isnull().sum()}")
        print(f"Distribution des espèces:\n{self.df['species'].value_counts()}")
        
        # Statistiques descriptives
        print(f"\nStatistiques descriptives:\n{self.df.describe()}")
        
    def visualize_data(self):
        """Créer des visualisations avancées."""
        print("\n📈 Création des visualisations...")
        
        # Configuration du style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Pairplot
        plt.subplot(3, 3, 1)
        sns.pairplot(self.df, hue='species', height=2)
        plt.title('Pairplot des caractéristiques')
        
        # 2. Matrice de corrélation
        plt.subplot(3, 3, 2)
        correlation_matrix = self.df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matrice de corrélation')
        
        # 3. Distribution des caractéristiques
        plt.subplot(3, 3, 3)
        self.df.select_dtypes(include=[np.number]).hist(bins=20, figsize=(12, 8))
        plt.suptitle('Distribution des caractéristiques')
        
        # 4. Boxplots par espèce
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for i, feature in enumerate(features):
            ax = axes[i//2, i%2]
            sns.boxplot(data=self.df, x='species', y=feature, ax=ax)
            ax.set_title(f'Distribution de {feature} par espèce')
        
        plt.tight_layout()
        plt.savefig('iris_data_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def prepare_data(self):
        """Préparer les données pour l'entraînement."""
        print("\n⚙️ Préparation des données...")
        
        # Séparer les caractéristiques et les cibles
        X = self.df.drop('species', axis=1)
        y = self.df['species']
        
        # Encoder les labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Normalisation des caractéristiques
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        
        print(f"✅ Données préparées: {X_train.shape[0]} échantillons d'entraînement, {X_test.shape[0]} de test")
        
    def initialize_models(self):
        """Initialiser les modèles de machine learning."""
        print("\n🤖 Initialisation des modèles...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Support Vector Machine': SVC(random_state=42, probability=True),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        print(f"✅ {len(self.models)} modèles initialisés")
        
    def train_and_evaluate(self):
        """Entraîner et évaluer tous les modèles."""
        print("\n🏋️ Entraînement et évaluation des modèles...")
        
        for name, model in self.models.items():
            print(f"\n📊 Évaluation de {name}...")
            
            # Entraînement
            model.fit(self.X_train, self.y_train)
            
            # Prédictions
            y_pred = model.predict(self.X_test)
            
            # Métriques
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Validation croisée
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            # Stocker les résultats
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
    def select_best_model(self):
        """Sélectionner le meilleur modèle basé sur la validation croisée."""
        print("\n🏆 Sélection du meilleur modèle...")
        
        best_score = 0
        best_name = ""
        
        for name, result in self.results.items():
            if result['cv_mean'] > best_score:
                best_score = result['cv_mean']
                best_name = name
                self.best_model = result['model']
        
        print(f"✅ Meilleur modèle: {best_name} (CV Score: {best_score:.4f})")
        return best_name, self.best_model
        
    def hyperparameter_tuning(self, model_name):
        """Optimiser les hyperparamètres du meilleur modèle."""
        print(f"\n🔧 Optimisation des hyperparamètres pour {model_name}...")
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'Support Vector Machine': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 1],
                'kernel': ['rbf', 'linear']
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        }
        
        if model_name in param_grids:
            model = self.models[model_name]
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"✅ Meilleurs paramètres: {grid_search.best_params_}")
            print(f"✅ Meilleur score: {grid_search.best_score_:.4f}")
            
            self.best_model = grid_search.best_estimator_
            return grid_search.best_estimator_
        else:
            print(f"⚠️ Pas d'optimisation définie pour {model_name}")
            return self.best_model
            
    def generate_detailed_report(self):
        """Générer un rapport détaillé des résultats."""
        print("\n📋 Génération du rapport détaillé...")
        
        # Prédictions du meilleur modèle
        y_pred = self.best_model.predict(self.X_test)
        
        # Rapport de classification
        print("\n📊 Rapport de classification:")
        target_names = self.label_encoder.classes_
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        
        # Matrice de confusion
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Matrice de confusion')
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe prédite')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Comparaison des modèles
        results_df = pd.DataFrame(self.results).T
        results_df = results_df[['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean']]
        
        plt.figure(figsize=(12, 8))
        results_df[['accuracy', 'precision', 'recall', 'f1_score']].plot(kind='bar')
        plt.title('Comparaison des performances des modèles')
        plt.ylabel('Score')
        plt.xlabel('Modèles')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results_df
        
    def save_model(self, filename='best_iris_model.pkl'):
        """Sauvegarder le meilleur modèle."""
        print(f"\n💾 Sauvegarde du modèle dans {filename}...")
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Modèle sauvegardé avec succès!")
        
    def predict_new_sample(self, sepal_length, sepal_width, petal_length, petal_width):
        """Prédire l'espèce d'un nouvel échantillon."""
        sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        sample_scaled = self.scaler.transform(sample)
        prediction = self.best_model.predict(sample_scaled)
        probability = self.best_model.predict_proba(sample_scaled)
        
        species = self.label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(probability)
        
        return species, confidence
        
    def run_complete_pipeline(self):
        """Exécuter le pipeline complet de classification."""
        print("🚀 Démarrage du pipeline de classification Iris...")
        
        # 1. Charger les données
        self.load_data()
        
        # 2. Explorer les données
        self.explore_data()
        
        # 3. Visualiser les données
        self.visualize_data()
        
        # 4. Préparer les données
        self.prepare_data()
        
        # 5. Initialiser les modèles
        self.initialize_models()
        
        # 6. Entraîner et évaluer
        self.train_and_evaluate()
        
        # 7. Sélectionner le meilleur modèle
        best_name, _ = self.select_best_model()
        
        # 8. Optimiser les hyperparamètres
        self.hyperparameter_tuning(best_name)
        
        # 9. Générer le rapport
        results_df = self.generate_detailed_report()
        
        # 10. Sauvegarder le modèle
        self.save_model()
        
        print("\n🎉 Pipeline terminé avec succès!")
        return results_df

def main():
    """Fonction principale."""
    classifier = IrisClassifier()
    results = classifier.run_complete_pipeline()
    
    # Test de prédiction
    print("\n🧪 Test de prédiction:")
    species, confidence = classifier.predict_new_sample(5.1, 3.5, 1.4, 0.2)
    print(f"Prédiction: {species} (Confiance: {confidence:.2%})")
    
    return classifier, results

if __name__ == "__main__":
    classifier, results = main()

