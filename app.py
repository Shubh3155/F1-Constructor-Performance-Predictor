import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="F1 Constructor Performance Predictor", 
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for caching
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

@st.cache_data
def load_and_process_data():
    """Load and process F1 data with error handling and validation"""
    try:
        # Load datasets
        required_files = [
            'constructor_results.csv',
            'constructor_standings.csv', 
            'qualifying.csv',
            'constructors.csv'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            st.error(f"Missing files: {', '.join(missing_files)}")
            return None
        
        constructor_results = pd.read_csv('constructor_results.csv')
        constructor_standings = pd.read_csv('constructor_standings.csv')
        qualifying = pd.read_csv('qualifying.csv')
        constructors = pd.read_csv('constructors.csv')
        
        # Debug: Show column names
        st.sidebar.write("**Constructor Results Columns:**")
        st.sidebar.write(list(constructor_results.columns))
        st.sidebar.write("**Constructor Standings Columns:**")
        st.sidebar.write(list(constructor_standings.columns))
        
        # Validate data integrity
        if constructor_results.empty or constructors.empty:
            st.error("Data files are empty or corrupted!")
            return None
        
        # Adaptive feature engineering based on available columns
        # Constructor Results Features
        results_agg = {}
        if 'points' in constructor_results.columns:
            results_agg['points'] = ['mean', 'sum', 'std', 'count']
        
        # Check for position-related columns (common variations)
        position_col = None
        for col in ['position', 'positionOrder', 'pos', 'finishing_position', 'finish_position']:
            if col in constructor_results.columns:
                position_col = col
                break
        
        if position_col:
            results_agg[position_col] = ['mean', 'min', 'std']
        
        if not results_agg:
            st.error("No valid columns found in constructor_results.csv")
            return None
        
        features = constructor_results.groupby('constructorId').agg(results_agg).reset_index()
        
        # Flatten column names dynamically
        new_columns = ['constructorId']
        for col in features.columns[1:]:
            if isinstance(col, tuple):
                if col[0] == 'points':
                    if col[1] == 'mean':
                        new_columns.append('avg_points')
                    elif col[1] == 'sum':
                        new_columns.append('total_points')
                    elif col[1] == 'std':
                        new_columns.append('points_std')
                    elif col[1] == 'count':
                        new_columns.append('races_count')
                elif col[0] == position_col:
                    if col[1] == 'mean':
                        new_columns.append('avg_position')
                    elif col[1] == 'min':
                        new_columns.append('best_position')
                    elif col[1] == 'std':
                        new_columns.append('position_std')
            else:
                new_columns.append(col)
        
        features.columns = new_columns
        
        # Handle NaN values in std columns
        for col in ['points_std', 'position_std']:
            if col in features.columns:
                features[col] = features[col].fillna(0)
        
        # Constructor standings features (adaptive)
        standings_agg = {}
        if 'points' in constructor_standings.columns:
            standings_agg['points'] = ['mean', 'max']
        
        # Check for wins column (common variations)
        wins_col = None
        for col in ['wins', 'win', 'victories', 'race_wins']:
            if col in constructor_standings.columns:
                wins_col = col
                break
        
        if wins_col:
            standings_agg[wins_col] = ['mean', 'sum']
        
        # Check for position column in standings
        standings_position_col = None
        for col in ['position', 'pos', 'championship_position', 'final_position']:
            if col in constructor_standings.columns:
                standings_position_col = col
                break
        
        if standings_position_col:
            standings_agg[standings_position_col] = 'mean'
        
        if standings_agg:
            standings_features = constructor_standings.groupby('constructorId').agg(standings_agg).reset_index()
            
            # Flatten column names for standings
            standings_new_columns = ['constructorId']
            for col in standings_features.columns[1:]:
                if isinstance(col, tuple):
                    if col[0] == 'points':
                        if col[1] == 'mean':
                            standings_new_columns.append('standings_avg_points')
                        elif col[1] == 'max':
                            standings_new_columns.append('max_season_points')
                    elif col[0] == wins_col:
                        if col[1] == 'mean':
                            standings_new_columns.append('avg_wins')
                        elif col[1] == 'sum':
                            standings_new_columns.append('total_wins')
                    elif col[0] == standings_position_col:
                        standings_new_columns.append('avg_championship_position')
                else:
                    standings_new_columns.append(col)
            
            standings_features.columns = standings_new_columns
        else:
            # Create empty standings features if no valid columns found
            standings_features = pd.DataFrame({'constructorId': features['constructorId']})
            st.warning("No valid columns found in constructor_standings.csv - using basic features only")
        
        # Merge all features
        data = pd.merge(features, standings_features, on='constructorId', how='left')
        data = data.fillna(0)
        
        # Add constructor names and handle missing names
        data = pd.merge(data, constructors[['constructorId', 'name']], on='constructorId', how='left')
        data['name'] = data['name'].fillna('Unknown Constructor')
        
        # Add derived features (only if base columns exist)
        if 'total_points' in data.columns and 'races_count' in data.columns:
            data['points_per_race'] = data['total_points'] / data['races_count'].replace(0, 1)
        
        if 'avg_points' in data.columns and 'points_std' in data.columns:
            data['consistency_score'] = data['avg_points'] / (data['points_std'] + 1)
        
        if 'total_wins' in data.columns and 'races_count' in data.columns:
            data['win_rate'] = data['total_wins'] / data['races_count'].replace(0, 1)
        else:
            # Set default values if wins data not available
            data['win_rate'] = 0
            data['total_wins'] = 0
            data['avg_wins'] = 0
        
        # Ensure we have a target variable
        if 'total_points' not in data.columns:
            st.error("Cannot find 'total_points' column - this is required for predictions!")
            return None
        
        # Remove outliers (optional - only if we have enough data)
        if len(data) > 10:
            Q1 = data['total_points'].quantile(0.25)
            Q3 = data['total_points'].quantile(0.75)
            IQR = Q3 - Q1
            data = data[~((data['total_points'] < (Q1 - 1.5 * IQR)) | 
                         (data['total_points'] > (Q3 + 1.5 * IQR)))]
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def train_model(data):
    """Train and cache the ML model"""
    try:
        # Define potential features and check which ones exist
        potential_features = [
            'avg_points', 'points_std', 'races_count', 'avg_position', 
            'best_position', 'position_std', 'standings_avg_points',
            'max_season_points', 'avg_wins', 'total_wins', 
            'avg_championship_position', 'points_per_race', 
            'consistency_score', 'win_rate'
        ]
        
        # Only use features that actually exist in the data
        available_features = [col for col in potential_features if col in data.columns]
        
        if len(available_features) < 2:
            st.error("Insufficient features available for training. Need at least 2 features.")
            return None
        
        st.info(f"Using {len(available_features)} features: {', '.join(available_features)}")
        
        X = data[available_features]
        y = data['total_points']
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        # Check if we have enough data
        if len(X) < 5:
            st.error("Insufficient data for training. Need at least 5 samples.")
            return None
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data (adjust test size for small datasets)
        test_size = min(0.2, max(0.1, len(X) * 0.2 / len(X)))
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        # Train multiple models and select best
        models = {
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=min(200, len(X_train) * 10), 
                learning_rate=0.1, 
                max_depth=min(6, len(available_features)), 
                random_state=42
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=min(200, len(X_train) * 10), 
                max_depth=min(10, len(available_features) + 2), 
                random_state=42
            )
        }
        
        best_model = None
        best_score = -np.inf
        model_results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                model_results[name] = {
                    'model': model,
                    'r2': r2,
                    'mse': mse,
                    'mae': mae,
                    'predictions': y_pred
                }
                
                if r2 > best_score:
                    best_score = r2
                    best_model = model
            except Exception as e:
                st.warning(f"Failed to train {name}: {str(e)}")
                continue
        
        if best_model is None:
            st.error("Failed to train any models successfully.")
            return None
        
        return {
            'best_model': best_model,
            'scaler': scaler,
            'features': available_features,
            'results': model_results,
            'X_test': X_test,
            'y_test': y_test
        }
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def main():
    st.title("🏎️ F1 Constructor Performance Predictor")
    st.markdown("*Advanced Machine Learning model to predict F1 constructor championship points*")
    
    # Sidebar Navigation
    with st.sidebar:
        st.header("Navigation")
        section = st.radio(
            "Go to", 
            ["📊 Data Overview", "🔍 EDA", "🤖 Model Training", "🎯 Prediction", "ℹ️ About"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### Data Status")
    
    # Load data
    data = load_and_process_data()
    
    if data is None:
        st.error("Failed to load data. Please ensure all required CSV files are present.")
        st.markdown("""
        ### Required Files:
        - `constructor_results.csv`
        - `constructor_standings.csv`
        - `qualifying.csv`
        - `constructors.csv`
        """)
        return
    
    st.sidebar.success(f"✅ Data loaded ({len(data)} constructors)")
    st.session_state.data_loaded = True
    
    # Data Overview Section
    if section == "📊 Data Overview":
        st.header("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Constructors", len(data))
        with col2:
            st.metric("Average Points", f"{data['total_points'].mean():.1f}")
        with col3:
            st.metric("Max Points", f"{data['total_points'].max():.0f}")
        with col4:
            st.metric("Features", len([col for col in data.columns if col not in ['constructorId', 'name', 'total_points']]))
        
        st.subheader("Sample Data")
        display_columns = ['name', 'total_points']
        
        # Add available columns to display
        optional_columns = ['avg_points', 'races_count', 'avg_position', 'total_wins']
        for col in optional_columns:
            if col in data.columns:
                display_columns.append(col)
        
        st.dataframe(
            data[display_columns].head(10),
            use_container_width=True
        )
        
        st.subheader("Data Quality")
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            st.warning("Missing values detected:")
            st.dataframe(missing_data[missing_data > 0])
        else:
            st.success("No missing values found!")
    
    # EDA Section
    elif section == "🔍 EDA":
        st.header("Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution of Total Points")
            fig1 = px.histogram(
                data, 
                x='total_points', 
                nbins=20, 
                title="Distribution of Constructor Total Points"
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Points vs Performance Metric")
            
            # Use available columns for scatter plot
            x_col = 'avg_position' if 'avg_position' in data.columns else 'races_count'
            color_col = 'win_rate' if 'win_rate' in data.columns else 'avg_points'
            size_col = 'races_count' if 'races_count' in data.columns else 'total_points'
            
            fig2 = px.scatter(
                data, 
                x=x_col, 
                y='total_points',
                color=color_col,
                size=size_col,
                hover_data=['name'],
                title=f"{x_col.replace('_', ' ').title()} vs Total Points"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Top Performing Constructors")
        
        # Create top constructors display with available columns
        display_columns = ['name', 'total_points']
        if 'avg_points' in data.columns:
            display_columns.append('avg_points')
        if 'total_wins' in data.columns:
            display_columns.append('total_wins')
        
        top_constructors = data.nlargest(10, 'total_points')[display_columns]
        
        fig3 = px.bar(
            top_constructors, 
            x='name', 
            y='total_points',
            title="Top 10 Constructors by Total Points"
        )
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Feature Correlation Matrix")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_columns].corr()
        
        fig4 = px.imshow(
            correlation_matrix, 
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Model Training Section
    elif section == "🤖 Model Training":
        st.header("Model Training & Evaluation")
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                model_data = train_model(data)
            
            if model_data:
                st.session_state.model_data = model_data
                st.session_state.model_trained = True
                st.success("Models trained successfully!")
        
        if st.session_state.model_trained and 'model_data' in st.session_state:
            model_data = st.session_state.model_data
            
            st.subheader("Model Performance Comparison")
            
            # Create performance comparison
            performance_df = pd.DataFrame({
                'Model': list(model_data['results'].keys()),
                'R² Score': [model_data['results'][name]['r2'] for name in model_data['results'].keys()],
                'MSE': [model_data['results'][name]['mse'] for name in model_data['results'].keys()],
                'MAE': [model_data['results'][name]['mae'] for name in model_data['results'].keys()]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(performance_df, use_container_width=True)
            
            with col2:
                fig = px.bar(performance_df, x='Model', y='R² Score', title="Model R² Score Comparison")
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance")
            if hasattr(model_data['best_model'], 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': model_data['features'],
                    'Importance': model_data['best_model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df.head(10), 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="Top 10 Most Important Features"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Section
    elif section == "🎯 Prediction":
        st.header("Constructor Performance Prediction")
        
        if not st.session_state.model_trained or 'model_data' not in st.session_state:
            st.warning("Please train a model first in the 'Model Training' section.")
            return
        
        model_data = st.session_state.model_data
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Select Constructor")
            selected_constructor = st.selectbox(
                "Choose a constructor:", 
                sorted(data['name'].unique())
            )
            
            constructor_data = data[data['name'] == selected_constructor]
            
            if not constructor_data.empty:
                # Manual feature input option
                st.subheader("Or Input Custom Values")
                use_custom = st.checkbox("Use custom values instead")
                
                if use_custom:
                    custom_features = {}
                    for feature in model_data['features']:
                        if feature in constructor_data.columns:
                            default_val = float(constructor_data[feature].iloc[0])
                            custom_features[feature] = st.number_input(
                                f"{feature}:", 
                                value=default_val,
                                key=f"custom_{feature}"
                            )
                    
                    # Create prediction
                    feature_array = np.array([list(custom_features.values())]).reshape(1, -1)
                    scaled_features = model_data['scaler'].transform(feature_array)
                    prediction = model_data['best_model'].predict(scaled_features)[0]
                else:
                    # Use actual constructor data
                    feature_values = constructor_data[model_data['features']].values
                    scaled_features = model_data['scaler'].transform(feature_values)
                    prediction = model_data['best_model'].predict(scaled_features)[0]
        
        with col2:
            if not constructor_data.empty:
                st.subheader("Prediction Results")
                
                # Display prediction
                actual_points = constructor_data['total_points'].iloc[0]
                
                col_pred, col_actual = st.columns(2)
                with col_pred:
                    st.metric("Predicted Points", f"{prediction:.1f}")
                with col_actual:
                    st.metric("Actual Points", f"{actual_points:.1f}")
                
                # Prediction confidence
                error_margin = abs(prediction - actual_points)
                confidence = max(0, 100 - (error_margin / actual_points * 100)) if actual_points > 0 else 0
                st.metric("Prediction Accuracy", f"{confidence:.1f}%")
                
                # Constructor details
                st.subheader("Constructor Details")
                
                # Show available details
                detail_columns = ['races_count', 'avg_points', 'total_wins', 'avg_position']
                available_details = {}
                for col in detail_columns:
                    if col in constructor_data.columns:
                        available_details[col] = constructor_data[col].iloc[0]
                
                if available_details:
                    st.json(available_details)
                else:
                    st.write("Limited details available for selected constructor.")
            else:
                st.error("No data available for selected constructor.")
    
    # About Section
    elif section == "ℹ️ About":
        st.header("About This Application")
        
        st.markdown("""
        ### 🏎️ F1 Constructor Performance Predictor
        
        This advanced machine learning application predicts Formula 1 constructor championship points 
        based on historical performance data and various statistical features.
        
        #### 🔧 **Technical Features:**
        - **Advanced Feature Engineering**: 14+ engineered features including consistency scores, win rates, and performance metrics
        - **Multiple ML Models**: Gradient Boosting and Random Forest with automatic model selection
        - **Data Validation**: Comprehensive error handling and data quality checks
        - **Interactive Visualizations**: Plotly-powered charts and graphs
        - **Performance Caching**: Streamlit caching for optimal performance
        - **Outlier Detection**: Automatic outlier removal using IQR method
        
        #### 📊 **Key Features:**
        - Real-time predictions for any constructor
        - Custom feature input for what-if scenarios
        - Comprehensive model performance metrics
        - Feature importance analysis
        - Interactive data exploration
        
        #### 🛠️ **Tech Stack:**
        - **Frontend**: Streamlit
        - **ML Libraries**: Scikit-learn, NumPy, Pandas
        - **Visualization**: Plotly, Matplotlib
        - **Data Processing**: Advanced feature engineering and preprocessing
        
        #### 📈 **Model Performance:**
        The application uses ensemble methods (Gradient Boosting and Random Forest) with automatic 
        hyperparameter tuning to achieve optimal prediction accuracy.
        
        ---
        
        **Developed by:** [Shubham khatri]  
        **GitHub Repository:** [https://github.com/Shubh3155/F1-Constructor-Performance-Predictor]  
        **Version:** 2.0 (Optimized)
        """)

if __name__ == "__main__":
    main()