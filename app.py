import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="F1 Constructor Performance Predictor",
    page_icon="🏎️",
    layout="wide"
)

# Cache data loading function
@st.cache_data
def load_and_preprocess_data():
    try:
        # Load datasets
        constructor_results = pd.read_csv('constructor_results.csv')
        constructor_standings = pd.read_csv('constructor_standings.csv')
        qualifying = pd.read_csv('qualifying.csv')
        constructors = pd.read_csv('constructors.csv')
        
        # Convert qualifying columns to numeric
        for col in ['q1', 'q2', 'q3']:
            qualifying[col] = pd.to_numeric(qualifying[col], errors='coerce')
        
        # Merge datasets
        merged_data = pd.merge(constructor_results, constructors[['constructorId', 'nationality']], on='constructorId')
        merged_data = pd.merge(merged_data, constructor_standings[['constructorId', 'raceId', 'wins']], on=['constructorId', 'raceId'])
        
        # Create features
        features = merged_data[['constructorId', 'nationality']].copy()
        features['avg_points'] = merged_data.groupby('constructorId')['points'].transform('mean')
        features['points_std'] = merged_data.groupby('constructorId')['points'].transform('std')
        features['win_rate'] = merged_data.groupby('constructorId')['wins'].transform('mean')
        
        # Nationality encoding
        le = LabelEncoder()
        features['nationality'] = le.fit_transform(features['nationality'])
        
        # Target variable
        y = merged_data['points']
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
        
        return features, y, merged_data, qualifying, constructors
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

@st.cache_resource
def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with optimized parameters
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions and metrics
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, rmse, r2, X_test_scaled, y_test, y_pred

def main():
    st.title("🏎️ Formula 1 Constructor Performance Predictor")
    st.markdown("### Predicting F1 Team Championship Points using Machine Learning")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Overview", "Data Analysis", "Model Performance", "Make Predictions"])
    
    # Load data
    features, y, merged_data, qualifying, constructors = load_and_preprocess_data()
    
    if features is None:
        st.error("Please upload the required CSV files to the same directory as this app.")
        st.info("Required files: constructor_results.csv, constructor_standings.csv, qualifying.csv, constructors.csv")
        return
    
    # Train model
    model, scaler, rmse, r2, X_test_scaled, y_test, y_pred = train_model(features, y)
    
    if page == "Overview":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Project Overview")
            st.write("""
            This project uses machine learning to predict Formula 1 constructor (team) 
            championship points based on historical performance data.
            
            **Key Features:**
            - Multi-dataset integration
            - Feature engineering (avg points, consistency, win rates)
            - Gradient Boosting Regressor
            - Comprehensive model evaluation
            """)
            
            st.subheader("Model Performance")
            st.metric("R² Score", f"{r2:.3f}")
            st.metric("RMSE", f"{rmse:.2f} points")
        
        with col2:
            st.subheader("Dataset Info")
            st.write(f"**Total Records:** {len(features):,}")
            st.write(f"**Features:** {len(features.columns)}")
            st.write(f"**Constructors:** {features['constructorId'].nunique()}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': features.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', 
                        title='Feature Importance', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Data Analysis":
        st.subheader("Exploratory Data Analysis")
        
        # Constructor nationality analysis
        if merged_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                nationality_performance = merged_data.groupby('nationality')['points'].mean().sort_values(ascending=False)[:10]
                fig = px.bar(x=nationality_performance.values, y=nationality_performance.index,
                           title='Average Points by Constructor Nationality', orientation='h')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Points distribution
                fig = px.histogram(merged_data, x='points', nbins=30,
                                 title='Distribution of Constructor Points')
                st.plotly_chart(fig, use_container_width=True)
        
        # Qualifying analysis
        if qualifying is not None:
            st.subheader("Qualifying Performance Analysis")
            qualifying_means = qualifying[['q1', 'q2', 'q3']].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=['Q1', 'Q2', 'Q3'], y=qualifying_means.values,
                                   mode='lines+markers', name='Average Time'))
            fig.update_layout(title='Average Qualifying Times Progression',
                            xaxis_title='Qualifying Session',
                            yaxis_title='Time (seconds)')
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Model Performance":
        st.subheader("Model Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted
            fig = px.scatter(x=y_test, y=y_pred, 
                           title='Actual vs Predicted Points',
                           labels={'x': 'Actual Points', 'y': 'Predicted Points'})
            # Add perfect prediction line
            min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                   mode='lines', name='Perfect Prediction', 
                                   line=dict(dash='dash', color='red')))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error distribution
            errors = y_pred - y_test
            fig = px.histogram(x=errors, nbins=30, title='Prediction Error Distribution')
            fig.update_layout(xaxis_title='Prediction Error', yaxis_title='Frequency')
            st.plotly_chart(fig, use_container_width=True)
        
        # Model metrics
        st.subheader("Detailed Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Root Mean Squared Error", f"{rmse:.2f}")
        with col2:
            st.metric("R² Score", f"{r2:.3f}")
        with col3:
            st.metric("Mean Absolute Error", f"{np.mean(np.abs(errors)):.2f}")
    
    elif page == "Make Predictions":
        st.subheader("Make Custom Predictions")
        st.write("Adjust the parameters below to see predicted constructor points:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            constructor_id = st.slider("Constructor ID", 1, 50, 1)
            nationality = st.slider("Nationality (encoded)", 0, 20, 5)
            avg_points = st.slider("Average Points", 0.0, 50.0, 10.0)
        
        with col2:
            points_std = st.slider("Points Standard Deviation", 0.0, 20.0, 5.0)
            win_rate = st.slider("Win Rate", 0.0, 1.0, 0.2)
        
        # Make prediction
        input_data = np.array([[constructor_id, nationality, avg_points, points_std, win_rate]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        st.subheader("Prediction Result")
        st.success(f"Predicted Constructor Points: **{prediction:.2f}**")
        
        # Show confidence interval (rough estimate)
        confidence = rmse * 1.96  # Approximate 95% confidence interval
        st.info(f"95% Confidence Interval: {prediction - confidence:.2f} to {prediction + confidence:.2f}")

if __name__ == "__main__":
    main()