import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="F1 Constructor Predictor", page_icon="🏎️")

st.title("🏎️ F1 Constructor Performance Predictor")
st.write("Machine Learning model to predict F1 constructor championship points")

# Check if data files exist
try:
    # Try to load data files
    constructor_results = pd.read_csv('constructor_results.csv')
    constructor_standings = pd.read_csv('constructor_standings.csv')
    qualifying = pd.read_csv('qualifying.csv')
    constructors = pd.read_csv('constructors.csv')
    
    st.success("✅ Data files loaded successfully!")
    
    # Show basic info
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Constructor Results", len(constructor_results))
    with col2:
        st.metric("Constructor Standings", len(constructor_standings))
    with col3:
        st.metric("Qualifying Records", len(qualifying))
    
    # Simple visualization
    st.subheader("Sample Visualization")
    if 'points' in constructor_results.columns:
        fig = px.histogram(constructor_results, x='points', title='Distribution of Constructor Points')
        st.plotly_chart(fig, use_container_width=True)
    
    # Simple model demo
    st.subheader("Model Demo")
    if st.button("Train Simple Model"):
        try:
            # Create simple features
            X = constructor_results[['constructorId']].fillna(0)
            y = constructor_results['points'].fillna(0)
            
            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Simple model
            model = GradientBoostingRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Prediction
            score = model.score(X_test, y_test)
            st.success(f"Model trained! R² Score: {score:.3f}")
            
        except Exception as e:
            st.error(f"Model training error: {e}")
    
except FileNotFoundError as e:
    st.error("❌ Data files not found!")
    st.write("Missing files:")
    st.write("- constructor_results.csv")
    st.write("- constructor_standings.csv") 
    st.write("- qualifying.csv")
    st.write("- constructors.csv")
    
    st.info("Upload these CSV files to your GitHub repository to enable full functionality.")
    
    # Show demo with sample data
    st.subheader("Demo with Sample Data")
    sample_data = pd.DataFrame({
        'Constructor': ['Mercedes', 'Red Bull', 'Ferrari', 'McLaren'],
        'Points': [450, 420, 380, 280],
        'Wins': [8, 7, 5, 2]
    })
    
    st.dataframe(sample_data)
    
    fig = px.bar(sample_data, x='Constructor', y='Points', title='Sample Constructor Points')
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Unexpected error: {e}")
    st.write("Please check your data files and try again.")

st.markdown("---")
st.markdown("**Project by:** Your Name | [GitHub](https://github.com/Shubh3155/F1-Constructor-Performance-Predictor)")