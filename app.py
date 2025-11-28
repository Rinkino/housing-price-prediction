import pickle
import warnings
import streamlit as st
import numpy as np
import pandas as pd

# Suppress sklearn version warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Housing Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Feature names that the model expects
FEATURE_NAMES = [
    'longitude',
    'latitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income',
    'ocean_proximity_INLAND',
    'ocean_proximity_ISLAND',
    'ocean_proximity_NEAR BAY',
    'ocean_proximity_NEAR OCEAN'
]

# Title and description
st.title("🏠 Housing Price Predictor")
st.markdown("Predict housing prices using machine learning based on property features and location.")

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Prediction", "Model Info", "Batch Prediction"])

if page == "Prediction":
    st.header("Single House Price Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Geographic Features")
        longitude = st.number_input("Longitude", value=-122.23, step=0.01)
        latitude = st.number_input("Latitude", value=37.88, step=0.01)
    
    with col2:
        st.subheader("Property Features")
        housing_median_age = st.number_input("Housing Median Age", value=41, step=1)
        total_rooms = st.number_input("Total Rooms", value=880, step=1)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Population & Households")
        total_bedrooms = st.number_input("Total Bedrooms", value=129, step=1)
        population = st.number_input("Population", value=322, step=1)
        households = st.number_input("Households", value=126, step=1)
    
    with col4:
        st.subheader("Economic & Location")
        median_income = st.number_input("Median Income", value=8.3252, step=0.1)
        
        st.subheader("Ocean Proximity")
        ocean_options = ["NEAR OCEAN", "INLAND", "NEAR BAY", "ISLAND"]
        selected_ocean = st.selectbox("Select location type:", ocean_options)
    
    # Create feature vector
    features = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity_INLAND': 1 if selected_ocean == "INLAND" else 0,
        'ocean_proximity_ISLAND': 1 if selected_ocean == "ISLAND" else 0,
        'ocean_proximity_NEAR BAY': 1 if selected_ocean == "NEAR BAY" else 0,
        'ocean_proximity_NEAR OCEAN': 1 if selected_ocean == "NEAR OCEAN" else 0,
    }
    
    # Make prediction
    if st.button("🔮 Predict Price", use_container_width=True):
        feature_values = np.array([features[f] for f in FEATURE_NAMES]).reshape(1, -1)
        prediction = model.predict(feature_values)[0]
        
        st.success("✅ Prediction Complete!")
        st.metric(label="Predicted House Price", value=f"${prediction:,.2f}")
        
        # Show input summary
        with st.expander("📊 Input Summary"):
            df = pd.DataFrame([features]).T
            df.columns = ["Value"]
            st.dataframe(df)

elif page == "Model Info":
    st.header("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Details")
        st.info(f"""
        **Model Type:** {type(model).__name__}
        
        **Number of Features:** {model.n_features_in_}
        
        **Intercept:** ${model.intercept_:,.2f}
        """)
    
    with col2:
        st.subheader("Feature Information")
        st.info(f"""
        **Features Used:**
        {chr(10).join([f"• {name}" for name in FEATURE_NAMES])}
        """)
    
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=False)
    
    st.dataframe(coef_df, use_container_width=True)
    
    st.bar_chart(coef_df.set_index('Feature')['Coefficient'])

elif page == "Batch Prediction":
    st.header("Batch Predictions")
    
    st.write("Upload a CSV file with housing data to get predictions for multiple houses.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.write("**Uploaded Data:**")
        st.dataframe(df)
        
        # Check if all required features are present
        missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
        
        if missing_features:
            st.error(f"❌ Missing features: {', '.join(missing_features)}")
        else:
            # Make predictions
            X = df[FEATURE_NAMES].values
            predictions = model.predict(X)
            
            # Add predictions to dataframe
            df['Predicted_Price'] = predictions
            
            st.success("✅ Predictions Complete!")
            st.dataframe(df, use_container_width=True)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
            # Statistics
            st.subheader("Prediction Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Price", f"${predictions.mean():,.2f}")
            with col2:
                st.metric("Min Price", f"${predictions.min():,.2f}")
            with col3:
                st.metric("Max Price", f"${predictions.max():,.2f}")
