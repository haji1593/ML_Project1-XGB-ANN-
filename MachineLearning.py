import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
import io

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Sidebar navigation
st.sidebar.title("🏥 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["Home", "Data Import & Overview", "Data Preprocessing", "Model Training",
     "Model Evaluation", "Prediction", "Interpretation & Conclusions"]
)


# Helper functions
# Add this with your other helper functions (after imports, before page logic)
def generate_dataset_summary(df):
    """Generate comprehensive dataset summary"""
    summary = {
        'Total Records': len(df),
        'Features': len(df.columns),
        'Missing Values': df.isnull().sum().sum(),
        'Diabetic Cases': df['Outcome'].sum(),
        'Non-Diabetic Cases': len(df) - df['Outcome'].sum(),
        'Missing Values (%)': f"{(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%"
    }
    return summary


def create_missing_values_heatmap(df):
    """Generate a clear heatmap of missing values"""
    missing_mask = df.isnull()

    fig = go.Figure(data=go.Heatmap(
        z=missing_mask.astype(int),
        x=df.columns,
        y=list(range(len(df))),
        colorscale=[[0, 'white'], [1, 'red']],
        showscale=True,
        name="Missing"
    ))

    fig.update_layout(
        title="Missing Values Heatmap",
        xaxis_title="Features",
        yaxis_title="Samples",
        height=600,
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=int(len(df) / 10)
        )
    )
    return fig


def analyze_correlations(df):
    """Analyze feature correlations with target"""
    corr_matrix = df.corr()['Outcome'].sort_values(ascending=False)
    strong_corr = corr_matrix[abs(corr_matrix) > 0.5]
    return strong_corr.drop('Outcome')


@st.cache_data
def load_sample_data():
    """Load sample Pima Indians Diabetes dataset"""
    # Sample data structure based on Pima Indians Diabetes Dataset
    np.random.seed(42)
    n_samples = 768

    data = {
        'Pregnancies': np.random.poisson(3, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples),
        'BloodPressure': np.random.normal(70, 20, n_samples),
        'SkinThickness': np.random.normal(20, 15, n_samples),
        'Insulin': np.random.normal(80, 115, n_samples),
        'BMI': np.random.normal(32, 7, n_samples),
        'DiabetesPedigreeFunction': np.random.exponential(0.5, n_samples),
        'Age': np.random.randint(21, 81, n_samples),
    }

    # Create outcome based on logical relationships
    outcome_prob = (
            (data['Glucose'] > 140) * 0.3 +
            (data['BMI'] > 30) * 0.2 +
            (data['Age'] > 50) * 0.2 +
            (data['Pregnancies'] > 5) * 0.1 +
            np.random.random(n_samples) * 0.2
    )

    data['Outcome'] = (outcome_prob > 0.5).astype(int)

    # Add some missing values
    df = pd.DataFrame(data)
    df.loc[df['Glucose'] < 50, 'Glucose'] = np.nan
    df.loc[df['BloodPressure'] < 20, 'BloodPressure'] = np.nan
    df.loc[df['SkinThickness'] < 5, 'SkinThickness'] = np.nan
    df.loc[df['Insulin'] < 10, 'Insulin'] = np.nan
    df.loc[df['BMI'] < 15, 'BMI'] = np.nan

    return df


def preprocess_data(df):
    """Preprocess the dataset"""
    df_processed = df.copy()

    # Handle missing values
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != 'Outcome':
            df_processed[col].fillna(df_processed[col].median(), inplace=True)

    # Remove negative values that don't make sense
    df_processed['Glucose'] = df_processed['Glucose'].clip(lower=0)
    df_processed['BloodPressure'] = df_processed['BloodPressure'].clip(lower=0)
    df_processed['SkinThickness'] = df_processed['SkinThickness'].clip(lower=0)
    df_processed['Insulin'] = df_processed['Insulin'].clip(lower=0)
    df_processed['BMI'] = df_processed['BMI'].clip(lower=0)

    return df_processed


# Add this helper function after your other helper functions (before the main page logic)
def create_engineered_features(df, feature_options):
    """
    Create engineered features based on selected options.

    Parameters:
    df (pd.DataFrame): Input dataframe
    feature_options (list): List of features to create

    Returns:
    pd.DataFrame: Dataframe with new features
    """
    df = df.copy()

    if "BMI Categories" in feature_options:
        df['BMI_Category'] = pd.cut(
            df['BMI'],
            bins=[-float('inf'), 18.5, 24.9, 29.9, float('inf')],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )

    if "Age Groups" in feature_options:
        df['Age_Group'] = pd.cut(
            df['Age'],
            bins=[-float('inf'), 20, 40, 60, float('inf')],
            labels=['Young', 'Adult', 'Middle_Aged', 'Senior']
        )

    if "Glucose Categories" in feature_options:
        df['Glucose_Category'] = pd.cut(
            df['Glucose'],
            bins=[-float('inf'), 70, 99, 125, float('inf')],
            labels=['Low', 'Normal', 'Prediabetes', 'Diabetes']
        )

    if "BP Categories" in feature_options:
        df['BP_Category'] = pd.cut(
            df['BloodPressure'],
            bins=[-float('inf'), 60, 80, 90, float('inf')],
            labels=['Low', 'Normal', 'PreHigh', 'High']
        )

    if "Insulin Sensitivity" in feature_options:
        df['Insulin_Sensitivity'] = df['Glucose'] / (df['Insulin'] + 1)

    return df
def train_models(X_train, X_test, y_train, y_test):
    """Train both ANN and XGBoost models"""
    models = {}

    # Train Neural Network
    ann = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2
    )
    ann.fit(X_train, y_train)
    models['Neural Network'] = ann

    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model

    return models


# Main content based on page selection
if page == "Home":
    st.markdown('<h1 class="main-header">🏥 Diabetes Prediction Application</h1>', unsafe_allow_html=True)

    st.markdown("""
    ## Welcome to the Diabetes Prediction App

    This application uses machine learning to predict diabetes risk based on health and lifestyle factors.

    ### 🎯 Purpose
    Diabetes is a major health concern affecting millions globally. Early detection and prevention are 
    critical for managing the disease and avoiding serious health complications.

    ### 🔬 What This App Does
    - **Data Analysis**: Explore and visualize diabetes-related health metrics
    - **Machine Learning**: Train and compare Neural Network and XGBoost models
    - **Prediction**: Make real-time diabetes risk predictions
    - **Insights**: Understand which factors most influence diabetes risk

    ### 📊 Features
    - Interactive data exploration with visualizations
    - Comprehensive model evaluation and comparison
    - User-friendly prediction interface
    - Feature importance analysis

    ### 🚀 Getting Started
    Use the navigation panel on the left to:
    1. Import and explore the dataset
    2. Preprocess the data
    3. Train machine learning models
    4. Evaluate model performance
    5. Make predictions
    6. Analyze results
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📈 **Data-Driven**\n\nBased on the famous Pima Indians Diabetes Dataset")
    with col2:
        st.success("🤖 **AI-Powered**\n\nUsing Neural Networks and XGBoost algorithms")
    with col3:
        st.warning("⚕️ **Healthcare Focus**\n\nDesigned for medical decision support")



# And replace everything until the next elif statement with this code:
elif page == "Data Import & Overview":
    st.markdown('<h2 class="sub-header">📊 Data Import & Overview</h2>', unsafe_allow_html=True)

    # File Upload Section
    st.subheader("1️⃣ Data Import")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Load Sample Dataset", key="load_sample"):
            with st.spinner("Loading sample dataset..."):
                st.session_state.data = load_sample_data()
                st.success("✅ Sample dataset loaded successfully!")

    with col2:
        uploaded_file = st.file_uploader(
            "📁 Upload Dataset (CSV/Excel)",
            type=["csv", "xlsx"],
            help="Upload a CSV or Excel file with diabetes-related features"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.data = pd.read_csv(uploaded_file)
                else:
                    st.session_state.data = pd.read_excel(uploaded_file)
                st.success("✅ Dataset uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")

    # Data Analysis Section
    if st.session_state.data is not None:
        df = st.session_state.data

        # Create tabs for different analyses
        tabs = st.tabs(["Summary", "Missing Values", "Column Details"])

        # Summary Tab
        with tabs[0]:
            st.subheader("2️⃣ Dataset Overview")

            # Display summary metrics
            summary = generate_dataset_summary(df)
            metrics = {
                "📊 Total Records": summary['Total Records'],
                "🎯 Features": summary['Features'],
                "❌ Missing Values": summary['Missing Values'],
                "🔴 Diabetic Cases": summary['Diabetic Cases'],
                "🟢 Non-Diabetic Cases": summary['Non-Diabetic Cases']
            }

            cols = st.columns(len(metrics))
            for col, (label, value) in zip(cols, metrics.items()):
                col.metric(label, value)

            # Data preview with tabs
            preview_tabs = st.tabs(["Head", "Tail", "Sample", "Statistics"])
            with preview_tabs[0]:
                st.dataframe(df.head(), use_container_width=True)
            with preview_tabs[1]:
                st.dataframe(df.tail(), use_container_width=True)
            with preview_tabs[2]:
                st.dataframe(df.sample(5), use_container_width=True)
            with preview_tabs[3]:
                st.dataframe(df.describe().round(2), use_container_width=True)

            # Correlation analysis
                # Find the correlation analysis part in the Summary Tab and replace it with:
                # Correlation analysis
            st.subheader("3️⃣ Feature Correlations")
            corr_matrix = df.corr()

                # Create correlation heatmap using plotly
            fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    text=np.round(corr_matrix, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False
                ))

            fig.update_layout(
                    title="Feature Correlation Matrix",
                    height=600,
                    width=800,
                    xaxis_title="Features",
                    yaxis_title="Features"
                )

            st.plotly_chart(fig, use_container_width=True)
        # Missing Values Tab
            # Find and replace the Missing Values Tab section with this code:
            # Missing Values Tab
        with tabs[1]:
            st.subheader("4️⃣ Missing Values Analysis")

                # Create a copy of the dataframe for missing value analysis
            df_missing = df.copy()

                # Replace 0s with NaN for features where 0 is not valid
            zero_as_null_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for feature in zero_as_null_features:
                df_missing[feature] = df_missing[feature].replace(0, np.nan)

                # Missing values summary
            missing_df = pd.DataFrame({
                    'Missing Values': df_missing.isnull().sum(),
                    'Missing (%)': (df_missing.isnull().sum() / len(df_missing) * 100).round(2)
                }).sort_values('Missing Values', ascending=False)

            col1, col2 = st.columns([2, 1])

            with col1:
                    # Missing values heatmap
                fig = go.Figure(data=go.Heatmap(
                        z=df_missing.isnull().astype(int),
                        x=df_missing.columns,
                        y=np.arange(len(df_missing)),
                        colorscale=[[0, 'white'], [1, 'red']],
                        showscale=True,
                        name="Missing"
                    ))
                fig.update_layout(
                        title="Missing Values Heatmap (Including 0s as Missing)",
                        xaxis_title="Features",
                        yaxis_title="Samples",
                        height=600,
                        yaxis=dict(
                            tickmode='linear',
                            tick0=0,
                            dtick=int(len(df_missing) / 10)
                        )
                    )
                st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.write("**Missing Values Summary:**")
                    st.dataframe(missing_df, use_container_width=True)

                    # Add explanation
                    st.info("""
                        **Note:** For medical features, zero values are treated as missing because they are physiologically implausible:
                        - Glucose level cannot be 0 mg/dL
                        - Blood Pressure cannot be 0 mmHg
                        - Skin Thickness cannot be 0 mm
                        - Insulin cannot be 0 μU/mL
                        - BMI cannot be 0
                        """)
        # Column Details Tab
                    # Find and replace the Column Details Tab section with this updated code:
                    # Column Details Tab
                    # Find and replace the Column Details Tab section:
                    # Column Details Tab
        # Find and replace the Column Details Tab section:
        # Column Details Tab
        with tabs[2]:
            st.subheader("5️⃣ Column Analysis")

            # Create a copy of the dataframe for analysis
            df_analysis = df.copy()

            # Replace 0s with NaN for features where 0 is not valid
            zero_as_null_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for feature in zero_as_null_features:
                if feature in df_analysis.columns:
                    df_analysis[feature] = df_analysis[feature].replace(0, np.nan)

            # Feature selection
            selected_cols = st.multiselect(
                "Select features to analyze:",
                options=df_analysis.columns.tolist(),
                default=df_analysis.select_dtypes(include=[np.number]).columns[:3].tolist()
            )

            if selected_cols:
                for col in selected_cols:
                    with st.expander(f"📊 {col} Analysis"):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            # Distribution plot
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(
                                x=df_analysis[col].dropna(),
                                nbinsx=30,
                                name=col
                            ))
                            fig.update_layout(
                                title=f"{col} Distribution",
                                xaxis_title=col,
                                yaxis_title="Count",
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            # Enhanced statistics
                            total_missing = df_analysis[col].isnull().sum()
                            zeros = (df_analysis[col] == 0).sum()

                            stats = pd.Series({
                                'Count': len(df_analysis[col].dropna()),
                                'Mean': df_analysis[col].mean(),
                                'Std': df_analysis[col].std(),
                                'Min': df_analysis[col].min(),
                                '25%': df_analysis[col].quantile(0.25),
                                'Median': df_analysis[col].median(),
                                '75%': df_analysis[col].quantile(0.75),
                                'Max': df_analysis[col].max(),
                                'Missing Values': total_missing,
                                'Zero Values': zeros,
                                'Total Invalid': total_missing + (zeros if col in zero_as_null_features else 0),
                                'Invalid (%)': (total_missing + (zeros if col in zero_as_null_features else 0)) / len(
                                    df_analysis) * 100,
                                'Unique Values': df_analysis[col].nunique()
                            })

                            # Format statistics
                            stats_df = pd.DataFrame({
                                'Metric': stats.index,
                                'Value': stats.values
                            })
                            st.dataframe(stats_df, use_container_width=True)

                            # Add warning if feature has significant missing/invalid values
                            invalid_percent = (total_missing + (zeros if col in zero_as_null_features else 0)) / len(
                                df_analysis) * 100
                            if invalid_percent > 5:
                                st.warning(f"⚠️ This feature has {invalid_percent:.1f}% invalid values " +
                                           "(including both missing values and zeros where not appropriate)")

                            # Add feature-specific insights
                            if col in zero_as_null_features:
                                st.info(
                                    f"ℹ️ Note: For {col}, zero values are considered invalid and should be treated as missing values.")
# Replace the Data Preprocessing section with this code:

# Replace the Data Preprocessing section:
# Replace the Data Preprocessing section:
elif page == "Data Preprocessing":
    st.markdown('<h2 class="sub-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("Please load data first from the 'Data Import & Overview' section.")
    else:
        # Create preprocessing tabs
        preprocessing_tabs = st.tabs([
            "1️⃣ Data Quality Check",
            "2️⃣ Missing Values",
            "3️⃣ Outlier Treatment",
            "4️⃣ Preview & Save"
        ])

        # Initialize dataframe in session state if not present
        if 'preprocessing_df' not in st.session_state:
            st.session_state.preprocessing_df = st.session_state.data.copy()

        # 1. Data Quality Check Tab
        with preprocessing_tabs[0]:
            st.subheader("Initial Data Quality Assessment")

            df = st.session_state.preprocessing_df

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Sample:**")
                st.dataframe(df.head(), use_container_width=True)

            with col2:
                st.write("**Data Info:**")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())

            # Check for zeros in medical features
            medical_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            zero_counts = pd.DataFrame({
                'Feature': medical_features,
                'Zero Count': [df[col].value_counts().get(0, 0) for col in medical_features],
                'Zero Percentage': [(df[col].value_counts().get(0, 0) / len(df)) * 100 for col in medical_features]
            })

            st.write("**Zero Values in Medical Features:**")
            st.dataframe(zero_counts.round(2), use_container_width=True)

        # 2. Missing Values Tab
        with preprocessing_tabs[1]:
            st.subheader("Handle Missing Values")

            # Option to convert zeros to NaN
            if st.checkbox("Convert zeros to missing values in medical features", value=True):
                for feature in medical_features:
                    df[feature] = df[feature].replace(0, np.nan)
                st.success("✅ Zeros converted to NaN")

            # Show missing value statistics
            missing_stats = pd.DataFrame({
                'Missing Values': df.isnull().sum(),
                'Missing (%)': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.write("**Missing Values Summary:**")
            st.dataframe(missing_stats, use_container_width=True)

            # Missing value treatment
            imputation_method = st.selectbox(
                "Select imputation method:",
                ["Median Imputation", "Mean Imputation", "KNN Imputation", "Drop Rows"]
            )

            if st.button("Apply Imputation"):
                with st.spinner("Applying imputation..."):
                    if imputation_method == "Median Imputation":
                        for col in df.select_dtypes(include=[np.number]).columns:
                            if col != 'Outcome':
                                df[col].fillna(df[col].median(), inplace=True)
                    elif imputation_method == "Mean Imputation":
                        for col in df.select_dtypes(include=[np.number]).columns:
                            if col != 'Outcome':
                                df[col].fillna(df[col].mean(), inplace=True)
                    elif imputation_method == "KNN Imputation":
                        from sklearn.impute import KNNImputer

                        imputer = KNNImputer(n_neighbors=5)
                        df_numeric = df.select_dtypes(include=[np.number])
                        df_numeric_imputed = pd.DataFrame(
                            imputer.fit_transform(df_numeric),
                            columns=df_numeric.columns
                        )
                        df = df_numeric_imputed
                    else:  # Drop Rows
                        df.dropna(inplace=True)

                    st.session_state.preprocessing_df = df
                    st.success("✅ Missing values handled successfully!")

        # 3. Outlier Treatment Tab
        with preprocessing_tabs[2]:
            st.subheader("Outlier Detection and Treatment")

            col1, col2 = st.columns(2)
            with col1:
                outlier_method = st.selectbox(
                    "Select outlier detection method:",
                    ["IQR Method", "Z-Score Method"]
                )

            with col2:
                outlier_treatment = st.selectbox(
                    "Select outlier treatment:",
                    ["Capping", "Remove"]
                )

            if st.button("Handle Outliers"):
                with st.spinner("Processing outliers..."):
                    df_before = df.copy()

                    if outlier_method == "IQR Method":
                        for col in df.select_dtypes(include=[np.number]).columns:
                            if col != 'Outcome':
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR

                                if outlier_treatment == "Capping":
                                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                                else:  # Remove
                                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

                    else:  # Z-Score Method
                        from scipy import stats

                        for col in df.select_dtypes(include=[np.number]).columns:
                            if col != 'Outcome':
                                z_scores = stats.zscore(df[col])
                                if outlier_treatment == "Capping":
                                    df[col] = df[col].mask(abs(z_scores) > 3, df[col].mean())
                                else:  # Remove
                                    df = df[abs(z_scores) < 3]

                    st.session_state.preprocessing_df = df

                    # Show outlier treatment results
                    st.write("**Records before outlier treatment:**", len(df_before))
                    st.write("**Records after outlier treatment:**", len(df))
                    st.success("✅ Outliers handled successfully!")

        # 4. Preview & Save Tab
        with preprocessing_tabs[3]:
            st.subheader("Preview and Save Processed Data")

            # Show before/after comparison
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Original Data Statistics:**")
                st.dataframe(st.session_state.data.describe().round(2),
                             use_container_width=True)

            with col2:
                st.write("**Processed Data Statistics:**")
                st.dataframe(df.describe().round(2), use_container_width=True)

            # Show sample of processed data
            st.write("**Processed Data Sample:**")
            st.dataframe(df.head(), use_container_width=True)

            # Save and download options
            col1, col2 = st.columns(2)

            with col1:
                if st.button("💾 Save Processed Data", type="primary"):
                    st.session_state.processed_data = df.copy()
                    st.success("✅ Processed data saved successfully! You can now proceed to Model Training.")

            with col2:
                if df is not None:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Processed Dataset",
                        data=csv,
                        file_name="processed_diabetes_data.csv",
                        mime="text/csv"
                    )
elif page == "Model Training":
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.warning("Please preprocess the data first.")
    else:
        df = st.session_state.processed_data

        st.subheader("1. Training Configuration")

        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        with col2:
            random_state = st.number_input("Random State", value=42)
        with col3:
            train_models_btn = st.button("🚀 Train Models", type="primary")

        if train_models_btn:
            # Prepare features and target
            X = df.drop('Outcome', axis=1)
            y = df['Outcome']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Store in session state
            st.session_state.X_train = X_train_scaled
            st.session_state.X_test = X_test_scaled
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.scaler = scaler
            st.session_state.feature_names = X.columns.tolist()

            with st.spinner("Training models..."):
                # Train Neural Network
                st.write("Training Neural Network...")
                ann = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=1000,
                    random_state=random_state,
                    early_stopping=True,
                    validation_fraction=0.2
                )
                ann.fit(X_train_scaled, y_train)

                # Train XGBoost
                st.write("Training XGBoost...")
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=random_state,
                    eval_metric='logloss'
                )
                xgb_model.fit(X_train_scaled, y_train)

                # Store models
                st.session_state.models = {
                    'Neural Network': ann,
                    'XGBoost': xgb_model
                }

            st.success("Models trained successfully!")

            # Display model information
            st.subheader("2. Model Information")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Neural Network Architecture:**")
                st.write(f"- Hidden Layers: {ann.hidden_layer_sizes}")
                st.write(f"- Iterations: {ann.n_iter_}")
                st.write(f"- Loss: {ann.loss_:.4f}")

            with col2:
                st.write("**XGBoost Parameters:**")
                st.write(f"- Estimators: {xgb_model.n_estimators}")
                st.write(f"- Max Depth: {xgb_model.max_depth}")
                st.write(f"- Learning Rate: {xgb_model.learning_rate}")

        # Feature importance (if models are trained)
        if st.session_state.models:
            st.subheader("3. Feature Importance")

            if 'XGBoost' in st.session_state.models:
                xgb_model = st.session_state.models['XGBoost']
                feature_names = st.session_state.feature_names

                importance = xgb_model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=True)

                fig = px.bar(feature_importance_df, x='Importance', y='Feature',
                             orientation='h', title='XGBoost Feature Importance')
                st.plotly_chart(fig, use_container_width=True)

elif page == "Model Evaluation":
    st.markdown('<h2 class="sub-header">📈 Model Evaluation</h2>', unsafe_allow_html=True)

    if not st.session_state.models:
        st.warning("Please train the models first.")
    else:
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        models = st.session_state.models

        st.subheader("1. Model Performance Metrics")

        # Calculate metrics for both models
        metrics_data = []
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred)
            }
            metrics_data.append(metrics)

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.round(4))

        # Visualize metrics comparison
        fig = go.Figure()

        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metrics_to_plot))

        for i, model_name in enumerate(models.keys()):
            values = [metrics_df[metrics_df['Model'] == model_name][metric].iloc[0]
                      for metric in metrics_to_plot]

            fig.add_trace(go.Bar(
                name=model_name,
                x=metrics_to_plot,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
            ))

        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Metrics',
            yaxis_title='Score',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Confusion Matrices
        st.subheader("2. Confusion Matrices")

        col1, col2 = st.columns(2)

        for i, (model_name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            fig = px.imshow(cm, text_auto=True, aspect="auto",
                            title=f'Confusion Matrix - {model_name}',
                            labels=dict(x="Predicted", y="Actual"))
            fig.update_xaxes(tickvals=[0, 1], ticktext=['Non-Diabetic', 'Diabetic'])
            fig.update_yaxes(tickvals=[0, 1], ticktext=['Non-Diabetic', 'Diabetic'])

            if i == 0:
                col1.plotly_chart(fig, use_container_width=True)
            else:
                col2.plotly_chart(fig, use_container_width=True)

        # ROC Curves
        st.subheader("3. ROC Curves")

        fig = go.Figure()

        for model_name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {roc_auc:.3f})',
                line=dict(width=2)
            ))

        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))

        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700, height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Model Comparison Conclusion
        st.subheader("4. Model Comparison Conclusion")

        best_model_idx = metrics_df['F1-Score'].idxmax()
        best_model = metrics_df.loc[best_model_idx, 'Model']
        best_f1 = metrics_df.loc[best_model_idx, 'F1-Score']

        st.write(f"**Best Performing Model:** {best_model}")
        st.write(f"**F1-Score:** {best_f1:.4f}")

        # Detailed analysis
        if best_model == 'Neural Network':
            st.info("""
            **Neural Network** performs better for this diabetes prediction task. 
            This suggests that the non-linear relationships in the data are well-captured 
            by the neural network's architecture.
            """)
        else:
            st.info("""
            **XGBoost** performs better for this diabetes prediction task. 
            This indicates that the ensemble approach and feature interactions 
            are effectively handled by the gradient boosting algorithm.
            """)

elif page == "Prediction":
    st.markdown('<h2 class="sub-header">🔮 Diabetes Risk Prediction</h2>', unsafe_allow_html=True)

    if not st.session_state.models:
        st.warning("Please train the models first.")
    else:
        st.subheader("Enter Patient Information")

        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
                glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
                blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=80)
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

            with col2:
                insulin = st.number_input("Insulin Level (μU/mL)", min_value=0, max_value=900, value=80)
                bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
                age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)

            predict_button = st.form_submit_button("🔍 Predict Diabetes Risk", type="primary")

        if predict_button:
            # Prepare input data
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, dpf, age]])

            # Scale the input data
            if st.session_state.scaler is not None:
                input_data_scaled = st.session_state.scaler.transform(input_data)
            else:
                input_data_scaled = input_data

            st.subheader("🎯 Prediction Results")

            # Make predictions with both models
            col1, col2 = st.columns(2)

            for i, (model_name, model) in enumerate(st.session_state.models.items()):
                prediction = model.predict(input_data_scaled)[0]
                probability = model.predict_proba(input_data_scaled)[0]

                # Display results
                with col1 if i == 0 else col2:
                    st.markdown(f"### {model_name}")

                    if prediction == 1:
                        st.error(f"⚠️ **HIGH RISK** of Diabetes")
                        st.write(f"Confidence: {probability[1]:.2%}")
                    else:
                        st.success(f"✅ **LOW RISK** of Diabetes")
                        st.write(f"Confidence: {probability[0]:.2%}")

                    # Probability bar
                    fig = go.Figure(go.Bar(
                        x=['Non-Diabetic', 'Diabetic'],
                        y=[probability[0], probability[1]],
                        marker_color=['green', 'red'],
                        text=[f'{probability[0]:.2%}', f'{probability[1]:.2%}'],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title=f'{model_name} - Risk Probability',
                        yaxis_title='Probability',
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Risk interpretation
            st.subheader("🩺 Risk Interpretation")

            avg_diabetes_prob = np.mean([
                st.session_state.models['Neural Network'].predict_proba(input_data_scaled)[0][1],
                st.session_state.models['XGBoost'].predict_proba(input_data_scaled)[0][1]
            ])

            if avg_diabetes_prob > 0.7:
                st.error("""
                **HIGH RISK**: The patient shows multiple risk factors for diabetes. 
                Immediate medical consultation is recommended.
                """)
            elif avg_diabetes_prob > 0.3:
                st.warning("""
                **MODERATE RISK**: Some risk factors are present. 
                Regular monitoring and lifestyle changes are advised.
                """)
            else:
                st.success("""
                **LOW RISK**: Current health indicators suggest low diabetes risk. 
                Maintain healthy lifestyle choices.
                """)

            # Risk factors analysis
            st.subheader("📊 Risk Factors Analysis")

            risk_factors = []
            if glucose > 140:
                risk_factors.append("High glucose level")
            if bmi > 30:
                risk_factors.append("High BMI (Obesity)")
            if age > 45:
                risk_factors.append("Advanced age")
            if pregnancies > 5:
                risk_factors.append("Multiple pregnancies")
            if blood_pressure > 90:
                risk_factors.append("High blood pressure")

            if risk_factors:
                st.write("**Identified Risk Factors:**")
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.write("**No major risk factors identified.**")

elif page == "Interpretation & Conclusions":
    st.markdown('<h2 class="sub-header">🔍 Interpretation & Conclusions</h2>', unsafe_allow_html=True)

    if not st.session_state.models or st.session_state.processed_data is None:
        st.warning("Please complete data preprocessing and model training first.")
    else:
        # Feature importance analysis
        st.subheader("1. Most Predictive Features")

        if 'XGBoost' in st.session_state.models:
            xgb_model = st.session_state.models['XGBoost']
            feature_names = st.session_state.feature_names

            importance = xgb_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)

            # Display top features
            st.write("**Top 5 Most Important Features for Diabetes Prediction:**")
            top_features = feature_importance_df.head(5)

            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                st.write(f"{i}. **{row['Feature']}** - Importance: {row['Importance']:.4f}")

            # Feature importance visualization
            fig = px.bar(feature_importance_df, x='Importance', y='Feature',
                         orientation='h', title='Feature Importance Analysis',
                         color='Importance', color_continuous_scale='viridis')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        # Model performance summary
        st.subheader("2. Model Performance Summary")

        if hasattr(st.session_state, 'X_test') and hasattr(st.session_state, 'y_test'):
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            models = st.session_state.models

            # Performance table
            performance_data = []
            for model_name, model in models.items():
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                performance_data.append({
                    'Model': model_name,
                    'Accuracy': f"{accuracy_score(y_test, y_pred):.4f}",
                    'Precision': f"{precision_score(y_test, y_pred):.4f}",
                    'Recall': f"{recall_score(y_test, y_pred):.4f}",
                    'F1-Score': f"{f1_score(y_test, y_pred):.4f}",
                    'AUC-ROC': f"{roc_auc:.4f}"
                })

            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df)

            # Model comparison insights
            st.subheader("3. Model Trade-offs Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **Neural Network Strengths:**
                - Captures complex non-linear relationships
                - Good at learning intricate patterns
                - Flexible architecture for feature interactions
                - Can model complex decision boundaries
                """)

                st.markdown("""
                **Neural Network Limitations:**
                - Black box model (less interpretable)
                - Requires more data for optimal performance
                - Prone to overfitting with small datasets
                - Longer training time
                """)

            with col2:
                st.markdown("""
                **XGBoost Strengths:**
                - High interpretability with feature importance
                - Handles missing values automatically
                - Built-in regularization prevents overfitting
                - Fast training and prediction
                - Robust to outliers
                """)

                st.markdown("""
                **XGBoost Limitations:**
                - May not capture very complex non-linear patterns
                - Requires hyperparameter tuning
                - Can be memory intensive for large datasets
                - Less effective with high-dimensional sparse data
                """)

        # Clinical insights
        st.subheader("4. Clinical Insights & Recommendations")

        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data

            # Statistical insights
            diabetic_data = df[df['Outcome'] == 1]
            non_diabetic_data = df[df['Outcome'] == 0]

            st.write("**Key Clinical Findings:**")

            # Compare means between groups
            insights = []
            for col in df.columns:
                if col != 'Outcome' and col in df.select_dtypes(include=[np.number]).columns:
                    diabetic_mean = diabetic_data[col].mean()
                    non_diabetic_mean = non_diabetic_data[col].mean()
                    diff_percent = ((diabetic_mean - non_diabetic_mean) / non_diabetic_mean) * 100

                    if abs(diff_percent) > 10:  # Only show significant differences
                        insights.append({
                            'Feature': col,
                            'Diabetic_Mean': diabetic_mean,
                            'Non_Diabetic_Mean': non_diabetic_mean,
                            'Difference_%': diff_percent
                        })

            insights_df = pd.DataFrame(insights).sort_values('Difference_%', key=abs, ascending=False)

            for _, row in insights_df.head(5).iterrows():
                direction = "higher" if row['Difference_%'] > 0 else "lower"
                st.write(f"- **{row['Feature']}**: {abs(row['Difference_%']):.1f}% {direction} in diabetic patients")

            # Visual comparison
            st.write("**Feature Comparison: Diabetic vs Non-Diabetic**")

            comparison_data = []
            for col in ['Glucose', 'BMI', 'Age', 'Insulin']:
                if col in df.columns:
                    comparison_data.extend([
                        {'Feature': col, 'Group': 'Diabetic', 'Value': diabetic_data[col].mean()},
                        {'Feature': col, 'Group': 'Non-Diabetic', 'Value': non_diabetic_data[col].mean()}
                    ])

            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                fig = px.bar(comparison_df, x='Feature', y='Value', color='Group',
                             barmode='group', title='Average Values: Diabetic vs Non-Diabetic Patients')
                st.plotly_chart(fig, use_container_width=True)

        # Practical recommendations
        st.subheader("5. Practical Implementation Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **For Healthcare Providers:**
            - Use this tool as a screening aid, not a diagnostic replacement
            - Focus on patients with glucose > 140 mg/dL for immediate attention
            - Monitor BMI and blood pressure as key modifiable risk factors
            - Consider family history (DPF) in risk assessment
            - Regular screening for patients with multiple risk factors
            """)

        with col2:
            st.markdown("""
            **For Patients:**
            - Maintain healthy BMI (< 25)
            - Monitor blood glucose levels regularly
            - Adopt a balanced, low-sugar diet
            - Engage in regular physical activity
            - Manage blood pressure through lifestyle changes
            - Regular health check-ups, especially if over 45
            """)

        # Model deployment considerations
        st.subheader("6. Deployment Considerations")

        st.markdown("""
        **Technical Recommendations:**
        - Regularly retrain models with new data to maintain accuracy
        - Implement data quality checks for input validation
        - Set up monitoring for model performance drift
        - Ensure compliance with healthcare data regulations (HIPAA)
        - Provide confidence intervals for predictions

        **Ethical Considerations:**
        - Ensure model fairness across different demographic groups
        - Provide clear explanations of model limitations to users
        - Maintain patient privacy and data security
        - Regular bias auditing and model validation
        - Human oversight for high-risk predictions
        """)

        # Final summary
        st.subheader("7. Executive Summary")

        st.info("""
        **Key Takeaways:**

        1. **Glucose level** is the most predictive feature for diabetes risk
        2. **BMI** and **Age** are significant modifiable and non-modifiable risk factors respectively
        3. Both Neural Network and XGBoost models show good performance, with trade-offs in interpretability vs complexity
        4. The application successfully identifies high-risk patients with high accuracy
        5. Early intervention based on these predictions can significantly improve patient outcomes

        **Next Steps:**
        - Validate models with external datasets
        - Conduct clinical trials for real-world validation
        - Integrate with electronic health record systems
        - Develop mobile applications for patient self-monitoring
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🏥 Diabetes Prediction Application | Built with Streamlit | For Educational and Research Purposes</p>
    <p><strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice.</p>
</div>
""", unsafe_allow_html=True)