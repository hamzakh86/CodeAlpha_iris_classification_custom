# Classification des Fleurs d'Iris - Projet Personnalisé

## Description
Ce projet implémente une solution complète de classification des fleurs d'Iris avec des améliorations par rapport au projet de référence. Il inclut plusieurs algorithmes de machine learning, une interface Streamlit interactive, et des visualisations avancées.

## Fonctionnalités
- 7 algorithmes de classification différents
- Interface web interactive avec Streamlit
- Visualisations avancées des données
- Optimisation automatique des hyperparamètres
- Métriques d'évaluation complètes
- Sauvegarde du meilleur modèle

## Installation

### Prérequis
- Python 3.7 ou plus récent
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Décompresser l'archive**
   ```bash
   tar -xzf iris_classification_custom.tar.gz
   cd iris_classification_custom
   ```

2. **Installer les dépendances**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly
   ```

3. **Exécuter l'entraînement du modèle (optionnel)**
   ```bash
   python iris_classification_enhanced.py
   ```

4. **Lancer l'application Streamlit**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Accéder à l'application**
   Ouvrez votre navigateur et allez à l'adresse affichée (généralement http://localhost:8501)

## Structure du projet

```
iris_classification_custom/
├── iris_classification_enhanced.py    # Script principal d'entraînement
├── streamlit_app.py                   # Application web Streamlit
├── iris_classification_main.ipynb     # Notebook Jupyter modifié
├── best_iris_model.pkl               # Modèle pré-entraîné
├── iris_data_visualization.png       # Visualisations des données
├── confusion_matrix.png              # Matrice de confusion
└── model_comparison.png              # Comparaison des modèles
```

## Utilisation

### Script Python
```bash
python iris_classification_enhanced.py
```
Ce script exécute le pipeline complet : chargement des données, entraînement des modèles, évaluation et sauvegarde.

### Application Streamlit
```bash
streamlit run streamlit_app.py
```
Lance l'interface web interactive pour :
- Explorer les données
- Faire des prédictions en temps réel
- Visualiser les résultats

### Notebook Jupyter
```bash
jupyter notebook iris_classification_main.ipynb
```
Version notebook pour l'exploration interactive.

## Modèles implémentés
1. Régression Logistique
2. K-Plus Proches Voisins (KNN)
3. Machine à Vecteurs de Support (SVM)
4. Naive Bayes
5. Arbre de Décision
6. Forêt Aléatoire
7. Gradient Boosting

## Métriques d'évaluation
- Accuracy (Précision)
- Precision (Précision par classe)
- Recall (Rappel)
- F1-Score
- Validation croisée

## Améliorations par rapport au projet original
- Ajout de nouveaux algorithmes (Random Forest, Gradient Boosting)
- Interface Streamlit interactive
- Optimisation automatique des hyperparamètres
- Visualisations avancées avec Plotly
- Métriques d'évaluation étendues
- Code modulaire et orienté objet
- Documentation complète

## Auteur
Hamza Khaled - Projet de stage en Data Science

## Licence
Ce projet est développé dans le cadre d'un stage académique.

