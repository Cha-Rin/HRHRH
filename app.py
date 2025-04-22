# app.py - Streamlit App for Iris Species Classifier
# Project2_GroupXX

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import io
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Project2_GroupXX",
    page_icon="🌸",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #26A69A;
    }
    .info-text {
        font-size: 1rem;
    }
    .highlight {
        color: #FF7043;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>Iris Species Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>A Business Application for Flower Classification</h3>", unsafe_allow_html=True)

# Create sidebar
st.sidebar.markdown("<h2 class='sub-header'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Home", "Data Explorer", "Model Performance", "Prediction"])

# Load data function with fallback options
@st.cache_data
def load_data():
    try:
        # Try loading directly (for local development)
        df = pd.read_csv('iris.csv')
    except:
        try:
            # If not available, load from sklearn
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                          columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
            # Convert numeric target to species names
            species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
            df['species'] = df['target'].map(species_map)
            df = df.drop('target', axis=1)
        except:
            # Last resort: create sample data
            data = {
                'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9],
                'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1],
                'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5],
                'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1],
                'species': ['setosa'] * 10
            }
            df = pd.DataFrame(data)
            # Add some versicolor and virginica samples
            df = pd.concat([df, pd.DataFrame({
                'sepal_length': [7.0, 6.4, 6.9, 5.5, 6.5],
                'sepal_width': [3.2, 3.2, 3.1, 2.3, 2.8],
                'petal_length': [4.7, 4.5, 4.9, 4.0, 4.6],
                'petal_width': [1.4, 1.5, 1.5, 1.3, 1.5],
                'species': ['versicolor'] * 5
            })], ignore_index=True)
            df = pd.concat([df, pd.DataFrame({
                'sepal_length': [6.3, 5.8, 7.1, 6.3, 6.5],
                'sepal_width': [3.3, 2.7, 3.0, 2.9, 3.0],
                'petal_length': [6.0, 5.1, 5.9, 5.6, 5.8],
                'petal_width': [2.5, 1.9, 2.1, 1.8, 2.2],
                'species': ['virginica'] * 5
            })], ignore_index=True)
    return df

# Load the data
df = load_data()

# Simple classifier function for when models can't be loaded
def simple_classifier(features):
    """A simple rule-based classifier for the iris dataset"""
    sepal_length, sepal_width, petal_length, petal_width = features
    
    if petal_length < 2.5:
        return "setosa", [0.95, 0.03, 0.02]
    elif petal_length < 5.0 and petal_width < 1.8:
        return "versicolor", [0.02, 0.90, 0.08]
    else:
        return "virginica", [0.01, 0.12, 0.87]

# Load models with fallback for demo mode
@st.cache_resource
def load_models():
    models = {}
    scaler = None
    best_model_name = "SVM"  # Default best model
    
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            # Try to create models by loading the dataset and training simple models
            df = load_data()
            X = df.drop('species', axis=1).values
            y = df['species'].values
            
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.svm import SVC
            
            # Create a simple SVM model
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            model = SVC(probability=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            models = {"SVM": model}
            return models, scaler, "SVM"
        
        # List all models in the models directory
        model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
        
        if not model_files:
            raise FileNotFoundError("No model files found in models directory")
        
        # Load each model
        for file in model_files:
            try:
                with open(f'models/{file}', 'rb') as f:
                    if file == 'scaler.pkl':
                        scaler = pickle.load(f)
                    elif file == 'best_model.pkl':
                        models['best_model'] = pickle.load(f)
                    else:
                        model_name = file.replace('.pkl', '').replace('_', ' ').title()
                        models[model_name] = pickle.load(f)
            except Exception as e:
                st.sidebar.warning(f"Error loading {file}: {str(e)}")
        
        # If best_model.pkl was loaded but no others, we need to know what type it is
        if 'best_model' in models and len(models) == 1:
            # Just use it with a generic name
            models["Best Model"] = models.pop('best_model')
            best_model_name = "Best Model"
    
    except Exception as e:
        st.sidebar.warning(f"Error loading models: {str(e)}")
        st.sidebar.info("Running in demo mode with simple classifier")
    
    # If no models were loaded, create a simple demo model
    if not models:
        st.sidebar.info("Using built-in simple classifier")
        
        # We'll use the simple_classifier function defined above for predictions
        # But here we create a placeholder model object for compatibility
        class SimpleModel:
            def predict(self, X):
                results = []
                for features in X:
                    species, _ = simple_classifier(features)
                    results.append(species)
                return np.array(results)
                
            def predict_proba(self, X):
                proba_results = []
                for features in X:
                    _, probs = simple_classifier(features)
                    proba_results.append(probs)
                return np.array(proba_results)
            
            @property
            def classes_(self):
                return np.array(['setosa', 'versicolor', 'virginica'])
                
        models = {"Simple Classifier": SimpleModel()}
        best_model_name = "Simple Classifier"
        
        # Create a simple scaler that just returns the input
        class IdentityScaler:
            def transform(self, X):
                return np.array(X)
                
        scaler = IdentityScaler()
    
    return models, scaler, best_model_name

# Load the models
try:
    models, scaler, best_model_name = load_models()
except Exception as e:
    st.error(f"Error initializing models: {str(e)}")
    # Create fallback models
    class DummyModel:
        def predict(self, X):
            return ["setosa"] * len(X)
        def predict_proba(self, X):
            return np.array([[1.0, 0.0, 0.0]] * len(X))
        @property
        def classes_(self):
            return np.array(['setosa', 'versicolor', 'virginica'])
    
    class DummyScaler:
        def transform(self, X):
            return np.array(X)
    
    models = {"Dummy": DummyModel()}
    scaler = DummyScaler()
    best_model_name = "Dummy"

# Home Page
if page == "Home":
    st.markdown("<h2 class='sub-header'>Project Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='info-text'>
        <p>This application demonstrates a <span class='highlight'>machine learning classification model</span> 
        for identifying iris flower species based on their measurements.</p>
        
        <p>The Iris dataset is a classic in the machine learning community, containing measurements 
        for 150 iris flowers of three different species:</p>
        <ul>
            <li><b>Setosa</b> - Found primarily in eastern Asia</li>
            <li><b>Versicolor</b> - Common in North America</li>
            <li><b>Virginica</b> - Native to eastern North America</li>
        </ul>
        
        <p><b>Business Application:</b> This classifier can help botanists, florists, and 
        researchers quickly identify iris species based on simple measurements, 
        saving time and improving accuracy in field research and commercial applications.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        try:
            # Display example plot
            species = df['species'].unique()
            colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
            
            fig, ax = plt.subplots(figsize=(5, 5))
            for species_name in species:
                subset = df[df['species'] == species_name]
                ax.scatter(subset['sepal_length'], subset['petal_length'], 
                          label=species_name, color=colors.get(species_name, 'gray'), alpha=0.7)
            
            ax.set_xlabel('Sepal Length (cm)')
            ax.set_ylabel('Petal Length (cm)')
            ax.set_title('Iris Species by Sepal and Petal Length')
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating plot: {str(e)}")
            # Show a placeholder image
            st.image("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.svg", 
                    caption="Iris flowers", width=300)
    
    # Feature explanation
    st.markdown("<h2 class='sub-header'>Feature Explanation</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-text'>
    <p>The model uses four key measurements of iris flowers to make predictions:</p>
    <ul>
        <li><b>Sepal Length</b>: The length of the outer parts of the flower (cm)</li>
        <li><b>Sepal Width</b>: The width of the outer parts of the flower (cm)</li>
        <li><b>Petal Length</b>: The length of the inner parts of the flower (cm)</li>
        <li><b>Petal Width</b>: The width of the inner parts of the flower (cm)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Project navigation
    st.markdown("<h2 class='sub-header'>Explore the Application</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 10px; background-color: #E3F2FD; border-radius: 5px;'>
            <h3>Data Explorer</h3>
            <p>View dataset statistics and visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 10px; background-color: #E8F5E9; border-radius: 5px;'>
            <h3>Model Performance</h3>
            <p>Check accuracy and other metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 10px; background-color: #FFF3E0; border-radius: 5px;'>
            <h3>Prediction</h3>
            <p>Try the model with your own measurements</p>
        </div>
        """, unsafe_allow_html=True)

# Data Explorer Page
elif page == "Data Explorer":
    st.markdown("<h2 class='sub-header'>Dataset Exploration</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Data Overview", "Visualizations", "Feature Analysis"])
    
    with tab1:
        st.markdown("### Dataset Head")
        st.dataframe(df.head())
        
        st.markdown("### Summary Statistics")
        st.dataframe(df.describe())
        
        st.markdown("### Species Distribution")
        species_counts = df['species'].value_counts().reset_index()
        species_counts.columns = ['Species', 'Count']
        
        try:
            fig = px.bar(species_counts, x='Species', y='Count', color='Species',
                        text='Count', title='Distribution of Iris Species')
            fig.update_layout(xaxis_title='Species', yaxis_title='Count')
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating bar chart: {str(e)}")
            st.table(species_counts)
    
    with tab2:
        st.markdown("### Feature Relationships")
        try:
            # Create individual scatter plots
            st.markdown("#### Scatter Plot Matrix")
            features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("X-axis feature", features, index=0)
            with col2:
                y_feature = st.selectbox("Y-axis feature", features, index=2)
                
            fig = px.scatter(df, x=x_feature, y=y_feature, color='species',
                           title=f'{y_feature} vs {x_feature} by Species')
            st.plotly_chart(fig)
            
            # Try to create pairplot
            st.markdown("#### Pair Plot")
            with st.spinner("Generating pairplot..."):
                fig, ax = plt.subplots(figsize=(10, 8))
                sns_plot = sns.pairplot(df, hue='species', height=2.5)
                st.pyplot(sns_plot.fig)
        except Exception as e:
            st.warning(f"Could not create some visualizations: {str(e)}")
        
        try:
            st.markdown("### 3D Scatter Plot")
            fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_length',
                             color='species', size='petal_width',
                             title='3D Scatter Plot of Iris Features')
            fig.update_layout(scene=dict(xaxis_title='Sepal Length',
                                      yaxis_title='Sepal Width',
                                      zaxis_title='Petal Length'))
            st.plotly_chart(fig)
        except Exception as e:
            st.warning(f"Could not create 3D plot: {str(e)}")
    
    with tab3:
        st.markdown("### Feature Distributions by Species")
        
        feature = st.selectbox("Select Feature", 
                              ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        
        try:
            fig = px.box(df, x='species', y=feature, color='species',
                       title=f'Distribution of {feature} by Species')
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating box plot: {str(e)}")
        
        st.markdown("### Feature Correlation Matrix")
        try:
            corr = df.drop('species', axis=1).corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis',
                           title='Feature Correlation Matrix')
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating correlation matrix: {str(e)}")
            st.dataframe(df.drop('species', axis=1).corr())

# Model Performance Page
elif page == "Model Performance":
    st.markdown("<h2 class='sub-")