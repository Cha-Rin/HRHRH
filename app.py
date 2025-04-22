import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set page configuration
st.set_page_config(
    page_title="Iris Dataset Visualization",
    page_icon="🌸",
    layout="wide"
)

# Function to load the dataset
@st.cache_data
def load_data():
    try:
        # Try to load from file
        data = pd.read_csv('iris.csv')
        return data
    except:
        # If not available, return a sample or error
        st.warning("Could not load the dataset. Using a sample dataset instead.")
        # Create a minimal sample
        return pd.DataFrame({
            'sepal_length': [5.1, 6.2, 7.3],
            'sepal_width': [3.5, 2.9, 2.9],
            'petal_length': [1.4, 4.3, 6.3],
            'petal_width': [0.2, 1.3, 1.8],
            'species': ['setosa', 'versicolor', 'virginica']
        })

# Load data
df = load_data()

# Main app title
st.title("🌸 Iris Dataset Visualization")
st.markdown("### Pairplot of Iris Features by Species")

# Create and display pairplot
fig = plt.figure(figsize=(12, 10))
sns.pairplot(df, hue='species', markers=["o", "o", "o"], 
             plot_kws={'alpha': 0.7, 's': 30}, 
             diag_kind='kde', 
             diag_kws={'alpha': 0.5, 'shade': True, 'bw_adjust': 0.8})
plt.tight_layout()
st.pyplot(fig)

# Add a brief explanation
st.markdown("""
### About This Visualization

This pairplot shows the relationships between the four features of the Iris dataset:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Each species is represented by a different color:
- Blue: Iris Setosa
- Orange: Iris Versicolor
- Green: Iris Virginica

The diagonal plots show the distribution of each feature for each species, while the scatter plots show the relationships between pairs of features.

This visualization clearly shows how the three species cluster differently based on their measurements, particularly for petal dimensions.
""")

# Footer
st.markdown("---")
st.markdown("Iris Flower Classification for Horticultural Business Decision Support")