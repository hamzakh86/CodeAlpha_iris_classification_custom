#!/usr/bin/env python3
"""
Application Streamlit pour la classification des fleurs d'Iris
Version personnalisée avec interface améliorée
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="Classification Iris - Hamza Khaled",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E8B57;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Charger les données Iris."""
    return sns.load_dataset('iris')

@st.cache_resource
def load_model():
    """Charger le modèle pré-entraîné."""
    try:
        with open('best_iris_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Modèle non trouvé. Veuillez d'abord exécuter l'entraînement.")
        return None

def main():
    """Fonction principale de l'application."""
    
    # En-tête
    st.markdown('<h1 class="main-header">🌸 Classification des Fleurs d\'Iris</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Version Personnalisée - Hamza Khaled")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choisir une page",
        ["🏠 Accueil", "📊 Exploration des données", "🔮 Prédiction", "📈 Analyse des modèles"]
    )
    
    # Charger les données
    df = load_data()
    model_data = load_model()
    
    if page == "🏠 Accueil":
        show_home_page(df)
    elif page == "📊 Exploration des données":
        show_data_exploration(df)
    elif page == "🔮 Prédiction":
        show_prediction_page(model_data)
    elif page == "📈 Analyse des modèles":
        show_model_analysis()

def show_home_page(df):
    """Afficher la page d'accueil."""
    st.markdown("## Bienvenue dans l'application de classification Iris!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Échantillons totaux", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Caractéristiques", len(df.columns) - 1)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Espèces", df['species'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Description du projet
    st.markdown("""
    ### 📋 Description du projet
    
    Cette application utilise des techniques d'apprentissage automatique pour classifier les fleurs d'Iris
    en trois espèces différentes basées sur leurs caractéristiques morphologiques:
    
    - **Iris Setosa** 🌺
    - **Iris Versicolor** 🌸  
    - **Iris Virginica** 🌼
    
    ### 🔬 Caractéristiques utilisées
    
    - **Longueur du sépale** (sepal_length)
    - **Largeur du sépale** (sepal_width)
    - **Longueur du pétale** (petal_length)
    - **Largeur du pétale** (petal_width)
    
    ### 🤖 Modèles implémentés
    
    - Régression Logistique
    - K-Plus Proches Voisins
    - Machine à Vecteurs de Support
    - Naive Bayes
    - Arbre de Décision
    - Forêt Aléatoire
    - Gradient Boosting
    """)

def show_data_exploration(df):
    """Afficher l'exploration des données."""
    st.markdown("## 📊 Exploration des données")
    
    # Aperçu des données
    st.subheader("Aperçu du dataset")
    st.dataframe(df.head(10))
    
    # Statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe())
    
    # Distribution des espèces
    st.subheader("Distribution des espèces")
    species_count = df['species'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(values=species_count.values, names=species_count.index,
                        title="Répartition des espèces")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(x=species_count.index, y=species_count.values,
                        title="Nombre d'échantillons par espèce")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Visualisations interactives
    st.subheader("Visualisations interactives")
    
    # Sélection des caractéristiques
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Axe X", features, index=0)
    with col2:
        y_axis = st.selectbox("Axe Y", features, index=2)
    
    # Scatter plot interactif
    fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color='species',
                           title=f"{x_axis} vs {y_axis}",
                           hover_data=features)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Matrice de corrélation
    st.subheader("Matrice de corrélation")
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    fig_heatmap = px.imshow(corr_matrix, 
                           text_auto=True, 
                           aspect="auto",
                           title="Matrice de corrélation des caractéristiques")
    st.plotly_chart(fig_heatmap, use_container_width=True)

def show_prediction_page(model_data):
    """Afficher la page de prédiction."""
    st.markdown("## 🔮 Prédiction d'espèce")
    
    if model_data is None:
        st.error("Modèle non disponible. Veuillez d'abord entraîner le modèle.")
        return
    
    st.markdown("### Entrez les caractéristiques de la fleur:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider("Longueur du sépale (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width = st.slider("Largeur du sépale (cm)", 2.0, 4.5, 3.0, 0.1)
    
    with col2:
        petal_length = st.slider("Longueur du pétale (cm)", 1.0, 7.0, 4.0, 0.1)
        petal_width = st.slider("Largeur du pétale (cm)", 0.1, 2.5, 1.2, 0.1)
    
    # Prédiction
    if st.button("🔍 Prédire l'espèce", type="primary"):
        sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        sample_scaled = model_data['scaler'].transform(sample)
        
        prediction = model_data['model'].predict(sample_scaled)
        probabilities = model_data['model'].predict_proba(sample_scaled)
        
        species = model_data['label_encoder'].inverse_transform(prediction)[0]
        confidence = np.max(probabilities)
        
        # Affichage des résultats
        st.markdown("---")
        st.markdown("### 🎯 Résultat de la prédiction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Espèce prédite", species)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Confiance", f"{confidence:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            # Emoji selon l'espèce
            emoji_map = {
                'setosa': '🌺',
                'versicolor': '🌸',
                'virginica': '🌼'
            }
            emoji = emoji_map.get(species, '🌸')
            st.markdown(f'<div style="font-size: 4rem; text-align: center;">{emoji}</div>', 
                       unsafe_allow_html=True)
        
        # Graphique des probabilités
        st.markdown("### 📊 Probabilités par espèce")
        species_names = model_data['label_encoder'].classes_
        prob_df = pd.DataFrame({
            'Espèce': species_names,
            'Probabilité': probabilities[0]
        })
        
        fig_prob = px.bar(prob_df, x='Espèce', y='Probabilité',
                         title="Distribution des probabilités")
        st.plotly_chart(fig_prob, use_container_width=True)

def show_model_analysis():
    """Afficher l'analyse des modèles."""
    st.markdown("## 📈 Analyse des modèles")
    
    st.markdown("""
    ### 🔬 Méthodologie
    
    Cette analyse compare plusieurs algorithmes d'apprentissage automatique:
    
    1. **Régression Logistique**: Modèle linéaire simple et interprétable
    2. **K-Plus Proches Voisins**: Classification basée sur la similarité
    3. **Machine à Vecteurs de Support**: Recherche de frontières optimales
    4. **Naive Bayes**: Approche probabiliste
    5. **Arbre de Décision**: Modèle basé sur des règles
    6. **Forêt Aléatoire**: Ensemble d'arbres de décision
    7. **Gradient Boosting**: Optimisation séquentielle
    
    ### 📊 Métriques d'évaluation
    
    - **Accuracy**: Pourcentage de prédictions correctes
    - **Precision**: Proportion de vrais positifs parmi les prédictions positives
    - **Recall**: Proportion de vrais positifs détectés
    - **F1-Score**: Moyenne harmonique de la précision et du rappel
    - **Cross-Validation**: Validation croisée à 5 plis
    
    ### 🏆 Avantages de cette approche
    
    - **Comparaison exhaustive** de multiple algorithmes
    - **Optimisation des hyperparamètres** pour le meilleur modèle
    - **Validation croisée** pour éviter le surapprentissage
    - **Métriques multiples** pour une évaluation complète
    - **Interface interactive** pour les prédictions en temps réel
    """)

if __name__ == "__main__":
    main()

