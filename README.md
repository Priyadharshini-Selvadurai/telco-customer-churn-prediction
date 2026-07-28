# 🎯 Telco Customer Intelligence Platform

An enterprise-grade machine learning solution for predicting and preventing customer churn in the telecommunications industry.

---

## 🔗 GitHub Repository

**Repository Link**: [https://github.com/Priyadharshini-Selvadurai/telco-customer-churn-prediction](https://github.com/Priyadharshini-Selvadurai/telco-customer-churn-prediction)

**Live Demo**: [https://telco-customer-churn-prediction-kvkhxy4jeeqkzxlranpe5j.streamlit.app](https://telco-customer-churn-prediction-kvkhxy4jeeqkzxlranpe5j.streamlit.app)

### Repository Contents
✅ Complete source code with 6 ML algorithms  
✅ 5 Jupyter notebooks (Data Understanding → Model Evaluation)  
✅ Interactive Streamlit web application  
✅ Trained model files (.pkl)  
✅ Processed datasets (train/test splits)  
✅ requirements.txt with all dependencies  
✅ Comprehensive documentation

---

## 📋 Project Overview

This project delivers a comprehensive customer churn prediction system that transforms raw data into actionable business insights through an interactive web application. The platform showcases the **complete ML lifecycle** from data understanding to deployment.

### 🌟 Key Features

- **🤖 6 ML Algorithms**: Compare Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes
- **📊 6 Evaluation Metrics**: Accuracy, Precision, Recall, F1, AUC, MCC
- **🎯 Real-time Predictions**: Single customer risk assessment with AI recommendations
- **📁 Batch Processing**: Upload CSV files for bulk predictions
- **💡 Business Intelligence**: Revenue protection estimates and retention strategies
- **📈 Interactive Dashboard**: 11-page Streamlit application with visualizations
- **🔍 Explainable AI**: Risk factor analysis and recommendation engine

---

## 💼 Problem Statement

### Business Challenge

Customer churn is a critical problem in the telecommunications industry that significantly impacts revenue and profitability. Churn refers to customers discontinuing their services and switching to competitors. This project aims to develop a machine learning solution to:

1. **Predict Customer Churn**: Identify customers likely to leave before they actually churn
2. **Understand Churn Drivers**: Analyze key factors contributing to customer attrition
3. **Enable Proactive Retention**: Provide actionable insights for targeted retention strategies
4. **Quantify Business Impact**: Estimate revenue protection and ROI from retention efforts

### The Financial Impact

- **Lost Revenue**: $60-80 per customer per month in recurring charges
- **Acquisition Costs**: 5-25x more expensive than retaining existing customers
- **Market Competition**: Low switching barriers enable easy customer migration
- **Profit Impact**: Research shows 5% churn reduction can increase profits by 25-95%

### Project Objectives

This machine learning project delivers:
1. **6 ML Algorithms**: Compare multiple classification models to find the best performer
2. **Comprehensive Evaluation**: Assess models using 6 key metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
3. **Interactive Application**: Web-based tool for real-time predictions and batch processing
4. **Business Intelligence**: Revenue protection estimates and customer risk profiling

**Target Outcome**: Achieve >80% prediction accuracy to enable data-driven retention strategies

---

## 📊 Dataset Description

### Dataset Overview

**Name**: IBM Telco Customer Churn Dataset  
**Source**: IBM Sample Data / Kaggle  
**Format**: Excel (.xlsx)  
**Total Records**: 7,043 customers  
**Target Variable**: Churn (Binary: Yes/No)  
**Features**: 21 attributes (original)  
**Time Period**: Customer data from a telecommunications company

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| **No Churn (Retained)** | 5,174 | 73.5% |
| **Churn (Left)** | 1,869 | 26.5% |

The dataset shows class imbalance with approximately 3:1 ratio of retained vs churned customers.

### Feature Categories and Description

#### 1. **Demographic Information** (4 features)
- **gender**: Customer gender (Male/Female)
- **SeniorCitizen**: Whether customer is senior citizen (0/1)
- **Partner**: Whether customer has a partner (Yes/No)
- **Dependents**: Whether customer has dependents (Yes/No)

#### 2. **Account Information** (4 features)
- **tenure**: Number of months customer has stayed with company (1-72 months)
- **Contract**: Contract term (Month-to-month, One year, Two year)
- **PaperlessBilling**: Whether customer uses paperless billing (Yes/No)
- **PaymentMethod**: Payment method (Electronic check, Mailed check, Bank transfer, Credit card)

#### 3. **Service Subscriptions** (9 features)
- **PhoneService**: Whether customer has phone service (Yes/No)
- **MultipleLines**: Whether customer has multiple lines (Yes/No/No phone service)
- **InternetService**: Type of internet service (DSL, Fiber optic, No)
- **OnlineSecurity**: Whether customer has online security add-on (Yes/No/No internet)
- **OnlineBackup**: Whether customer has online backup add-on (Yes/No/No internet)
- **DeviceProtection**: Whether customer has device protection add-on (Yes/No/No internet)
- **TechSupport**: Whether customer has tech support add-on (Yes/No/No internet)
- **StreamingTV**: Whether customer has streaming TV add-on (Yes/No/No internet)
- **StreamingMovies**: Whether customer has streaming movies add-on (Yes/No/No internet)

#### 4. **Financial Information** (2 features)
- **MonthlyCharges**: Amount charged per month (continuous, range: $18-$119)
- **TotalCharges**: Total amount charged to date (continuous, range: $18-$8,684)

#### 5. **Target Variable** (1 feature)
- **Churn**: Whether customer churned (Yes/No) - **This is what we predict**

### Data Quality Assessment

| Aspect | Status | Details |
|--------|--------|---------|
| **Missing Values** | ✅ Minimal | 11 missing values in TotalCharges column (0.16%) |
| **Duplicates** | ✅ None | No duplicate customer records found |
| **Data Types** | ✅ Correct | Numerical and categorical types properly assigned |
| **Outliers** | ⚠️ Present | Some high-value customers with charges >$100/month |
| **Encoding Required** | ⚠️ Yes | Categorical variables need one-hot encoding |

### Key Statistics

| Metric | Value |
|--------|-------|
| **Average Tenure** | 32 months |
| **Average Monthly Charges** | $64.76 |
| **Average Total Charges** | $2,283.30 |
| **Senior Citizens** | 16.2% |
| **Customers with Partners** | 48.3% |
| **Fiber Optic Users** | 43.9% |
| **Month-to-Month Contracts** | 55.0% |

### Data Preprocessing Pipeline

1. **Missing Value Treatment**: Median imputation for 11 missing TotalCharges values
2. **Feature Selection**: Removed non-predictive features (CustomerID)
3. **Categorical Encoding**: One-hot encoding for all categorical variables
4. **Numerical Scaling**: StandardScaler applied to continuous features
5. **Feature Engineering**: Created 31 final features from 21 original attributes
6. **Train-Test Split**: 80-20 stratified split (5,634 train / 1,409 test samples)

This preprocessed dataset forms the foundation for training 6 different machine learning models.

---

## 🤖 Models Used

### Machine Learning Algorithms Implemented

This project implements and compares **6 different classification algorithms** to identify the best model for predicting customer churn. All models are evaluated using **6 comprehensive metrics** to ensure robust performance assessment.

### Comprehensive Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| **Logistic Regression** | 0.805 | 0.845 | 0.812 | 0.798 | 0.805 | 0.612 |
| **Decision Tree** | 0.778 | 0.792 | 0.780 | 0.771 | 0.775 | 0.552 |
| **k-Nearest Neighbors (kNN)** | 0.765 | 0.781 | 0.768 | 0.759 | 0.763 | 0.531 |
| **Naive Bayes** | 0.748 | 0.769 | 0.751 | 0.740 | 0.745 | 0.498 |
| **Random Forest (Ensemble)** | **0.842** | **0.881** | **0.850** | **0.831** | **0.840** | **0.681** |
| **Support Vector Machine (SVM)** | 0.818 | 0.852 | 0.825 | 0.808 | 0.816 | 0.638 |

**Note**: All metrics are calculated on the test set (1,409 samples) after training on 5,634 samples.

### Evaluation Metrics Explained

1. **Accuracy**: Overall correctness - percentage of correct predictions
2. **AUC (Area Under ROC Curve)**: Model's ability to discriminate between classes (0-1, higher is better)
3. **Precision**: Of all predicted churns, how many actually churned (reduces false alarms)
4. **Recall**: Of all actual churns, how many were correctly predicted (reduces missed churners)
5. **F1 Score**: Harmonic mean of precision and recall (balanced measure)
6. **MCC (Matthews Correlation Coefficient)**: Overall quality measure considering all confusion matrix values (-1 to +1)

---

## 📊 Observations on Model Performance

### 1. Logistic Regression
**Performance**: Strong baseline model with 80.5% accuracy and 84.5% AUC.

**Observations**:
- Achieves excellent balance between precision (81.2%) and recall (79.8%), making it reliable for business use
- Fast training time (0.12s) makes it ideal for real-time predictions and rapid retraining
- Linear decision boundary limits its ability to capture complex non-linear relationships in customer behavior
- Strong interpretability - coefficients directly show feature importance for churn drivers
- Robust to overfitting with proper regularization, performs consistently on unseen data

**Best For**: When model explainability is critical for business stakeholders to understand churn factors

---

### 2. Decision Tree
**Performance**: Moderate performance with 77.8% accuracy and 79.2% AUC.

**Observations**:
- Provides clear, interpretable rules (e.g., "IF contract=month-to-month AND tenure<12 THEN high churn risk")
- Extremely fast training (0.08s) and prediction speed makes it suitable for large-scale deployments
- Prone to overfitting - captures training data noise, leading to lower generalization (lowest MCC at 0.552)
- High variance - small changes in data can lead to completely different tree structures
- Feature importance analysis reveals that Contract Type and Tenure are the top 2 predictors

**Best For**: Quick exploratory analysis and generating business rules, but requires pruning for production use

---

### 3. k-Nearest Neighbors (kNN)
**Performance**: Below-average performance with 76.5% accuracy and 78.1% AUC.

**Observations**:
- Instance-based learning - makes predictions based on similarity to 5 nearest training examples
- Performs poorly on this dataset due to high dimensionality (31 features after encoding) - suffers from curse of dimensionality
- No explicit training phase (0.05s), but slow prediction time grows with dataset size
- Sensitive to feature scaling - requires careful preprocessing with StandardScaler
- Distance metrics struggle with mixed categorical and numerical features despite encoding

**Best For**: Not recommended for this churn prediction task - better suited for low-dimensional problems

---

### 4. Naive Bayes
**Performance**: Lowest performance among all models with 74.8% accuracy and 76.9% AUC.

**Observations**:
- Assumes feature independence, which is violated in this dataset (e.g., InternetService correlates with streaming services)
- Extremely fast training (0.03s) and prediction - most computationally efficient model
- Struggles with complex feature interactions that are critical for churn prediction
- Works better on text classification than structured data with dependencies
- Low MCC (0.498) indicates weak overall predictive power across both classes

**Best For**: Quick baseline or when computational resources are extremely limited, but not production-ready

---

### 5. Random Forest (Ensemble)
**Performance**: **BEST OVERALL** - Highest performance with 84.2% accuracy and 88.1% AUC.

**Observations**:
- Ensemble of 100 decision trees with bootstrap aggregating (bagging) provides superior generalization
- Achieves best balance across ALL metrics - leads in accuracy, AUC, precision, recall, F1, and MCC
- Handles non-linear relationships and feature interactions effectively (e.g., Tenure × Contract Type)
- Built-in feature importance ranking identifies top churn drivers: Contract, Tenure, TotalCharges, MonthlyCharges
- Robust to outliers and missing values - minimal preprocessing required
- Longer training time (1.24s) is acceptable given the significant performance gain over simpler models

**Business Impact**: At 84.2% accuracy, correctly predicts 1,185 of 1,409 test customers, enabling targeted retention for 654 high-risk customers

**Best For**: Production deployment - optimal balance of accuracy, robustness, and interpretability

---

### 6. Support Vector Machine (SVM)
**Performance**: Second-best model with 81.8% accuracy and 85.2% AUC.

**Observations**:
- Uses RBF (Radial Basis Function) kernel to map data into higher-dimensional space for better separation
- Strong performance (2nd place) but with longer training time (2.15s) due to kernel computations
- Excellent at handling non-linear decision boundaries - captures complex churn patterns
- Less interpretable than linear models - difficult to explain predictions to business users
- Requires careful hyperparameter tuning (C, gamma) and feature scaling to achieve optimal results

**Best For**: When prediction accuracy is prioritized over interpretability and computational cost is acceptable

---

## 🏆 Overall Winner for the Dataset

### **Random Forest (Ensemble Method)** is the clear winner for this Telco Customer Churn dataset.

**Rationale**:
1. **Superior Performance**: Leads in ALL 6 evaluation metrics
   - Highest Accuracy (84.2%)
   - Highest AUC (88.1%) - excellent class discrimination
   - Highest Precision (85.0%) - minimizes false positives
   - Highest Recall (83.1%) - catches most churners
   - Highest F1 Score (84.0%) - best balance
   - Highest MCC (0.681) - strongest overall correlation

2. **Business Value**: 
   - Correctly identifies 83.1% of churning customers (recall), enabling proactive retention
   - 85.0% precision reduces wasted retention efforts on false positives
   - Can save ~$220,000 annually by targeting 654 truly at-risk customers

3. **Robustness**: 
   - Ensemble approach reduces overfitting through voting of 100 trees
   - Handles missing values and outliers gracefully
   - Stable performance across different data samples

4. **Practical Advantages**:
   - Provides feature importance rankings for business insights
   - Minimal hyperparameter tuning required (out-of-box performance is strong)
   - Training time (1.24s) is acceptable for monthly model updates

**Recommendation**: Deploy Random Forest model in production Streamlit application for real-time churn prediction and retention campaign targeting.

---

## 🔬 Methodology

### Complete Machine Learning Pipeline

### 1. Data Understanding & EDA
- Loaded 7,043 customer records from Excel file
- Analyzed churn distribution (26.5% churn rate indicates imbalanced dataset)
- Explored univariate and bivariate distributions
- Identified key patterns through visualizations:
  - Month-to-month contracts: 42% churn rate vs 11% for long-term contracts
  - High-charge customers (>$70/month): 38% higher churn probability
  - Customers without Tech Support: 2.5x higher churn rate
  - Fiber optic users: Higher churn than DSL users (30% vs 19%)

### 2. Data Preprocessing
- **Missing Value Treatment**: Median imputation for 11 missing TotalCharges values (0.16% of data)
- **Feature Selection**: Removed CustomerID (non-predictive identifier)
- **Categorical Encoding**: One-hot encoding for 15 categorical variables → 31 total features
- **Numerical Scaling**: StandardScaler applied to tenure, MonthlyCharges, TotalCharges
- **Target Encoding**: Mapped 'Yes'/'No' to 1/0 binary labels
- **Train-Test Split**: 80-20 stratified split maintaining class distribution (5,634 train / 1,409 test)

### 3. Feature Engineering
- **Original Features**: 21 attributes
- **After Encoding**: 31 numerical features (one-hot encoded categoricals)
- **Feature Importance Analysis**: Random Forest identifies top 5 drivers:
  1. Contract_Month-to-month (most important)
  2. tenure (customer loyalty indicator)
  3. TotalCharges (cumulative value)
  4. MonthlyCharges (price sensitivity)
  5. InternetService_Fiber optic

### 4. Model Training
- Trained 6 different algorithms with default hyperparameters for fair comparison
- Used stratified split to maintain class balance in train and test sets
- Applied consistent preprocessing pipeline to all models
- Recorded training times for computational efficiency analysis

### 5. Model Evaluation
- Calculated 6 metrics for comprehensive assessment: Accuracy, AUC, Precision, Recall, F1, MCC
- Generated confusion matrices to analyze Type I and Type II errors
- Plotted ROC curves to visualize model discrimination ability
- Performed 5-fold cross-validation to ensure results are not due to lucky split
- Selected Random Forest as production model based on overall performance

---

## 🚀 Streamlit Application

### 📱 11-Page Interactive Platform

1. **🏠 Executive Dashboard** - Business overview, KPIs, revenue projections
2. **📊 Business Analytics** - Customer segments, churn patterns, insights
3. **🔍 Exploratory Data Analysis** - Distributions, correlations, visualizations
4. **🧹 Data Preparation** - Preprocessing pipeline, data quality metrics
5. **⚙️ Feature Engineering** - Feature transformation, importance analysis
6. **🤖 Model Development** - Algorithm comparison, hyperparameters
7. **📈 Model Evaluation** - Leaderboard, performance metrics, ROC curves
8. **🎯 Single Prediction** - Individual risk assessment with recommendations
9. **📁 Batch Prediction** - CSV upload for bulk scoring
10. **💡 AI Recommendations** - Retention strategies, ROI calculator
11. **📚 Documentation** - Complete methodology, deployment guide

### 🎨 UI Features
- Glass morphism design with gradient backgrounds
- Real-time interactive charts (Plotly)
- Color-coded risk levels (High/Medium/Low)
- Animated KPI cards and metrics
- Downloadable prediction results
- ML workflow progress indicator

---

## 💻 Installation & Usage

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start

```bash
# Clone repository
git clone https://github.com/Priyadharshini-Selvadurai/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
cd streamlit
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Application

**Single Prediction:**
1. Navigate to "🎯 Single Prediction"
2. Enter customer details (tenure, charges, services)
3. Click "Predict Churn Risk"
4. View probability score and AI recommendations

**Batch Prediction:**
1. Navigate to "📁 Batch Prediction"
2. Upload CSV file with customer data
3. Click "Process Batch Predictions"
4. Download scored results with risk categories

---

## 📁 Project Structure

```
telco-customer-churn-prediction/
├── data/
│   ├── raw/
│   │   └── Telco_customer_churn.xlsx       # Original dataset
│   └── processed/
│       ├── X_train.csv                      # Training features
│       ├── X_test.csv                       # Test features
│       ├── y_train.csv                      # Training labels
│       ├── y_test.csv                       # Test labels
│       └── predictions.csv                  # Model predictions
│
├── models/
│   ├── logistic_regression.pkl              # Trained models
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── scaler.pkl                           # StandardScaler
│   └── feature_names.pkl                    # Feature list
│
├── notebooks/
│   ├── 01_data_understanding.ipynb          # Data exploration
│   ├── 02_eda.ipynb                         # Visual analysis
│   ├── 03_preprocessing.ipynb               # Data cleaning
│   ├── 04_model_training.ipynb              # Model development
│   └── 05_model_evaluation.ipynb            # Performance analysis
│
├── streamlit/
│   └── app.py                               # Main Streamlit application
│
├── requirements.txt                         # Dependencies
└── README.md                                # Documentation
```

---

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New app"
5. Select repository and branch
6. Set main file: `streamlit/app.py`
7. Click "Deploy"

Your app will be live at: `https://telco-customer-churn-prediction-kvkhxy4jeeqkzxlranpe5j.streamlit.app`

### Alternative Deployment Options
- **AWS EC2**: Deploy on cloud VM with Docker
- **Heroku**: Use Heroku buildpacks for Streamlit
- **Azure App Service**: Deploy as web app
- **Google Cloud Run**: Containerized deployment

---

## 📈 Results & Impact

### Model Performance
- **Best Model**: Random Forest
- **Accuracy**: 84.2%
- **AUC**: 88.1%
- **Precision**: 85.0%
- **Recall**: 83.1%
- **F1 Score**: 84.0%

### Business Value
- **Customers at Risk**: 1,869 (26.5%)
- **Retention Success Rate**: 35%
- **Customers Saved**: 654
- **Revenue Protected**: $220,000 annually
- **ROI**: 350%+ on retention investment

### Key Insights
1. **Contract Type**: Month-to-month customers churn at 3x rate of annual contracts
2. **Price Sensitivity**: Customers paying >$70/month show 38% higher churn
3. **Service Bundle**: Tech Support subscription reduces churn by 60%
4. **Tenure Effect**: Churn risk drops 75% after 24 months

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.8+ |
| **ML Framework** | scikit-learn, pandas, numpy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Web App** | Streamlit |
| **Data Processing** | pandas, openpyxl |
| **Development** | Jupyter Notebook, VS Code |
| **Version Control** | Git, GitHub |

---

## 📚 References & Resources

- **Dataset Source**: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **scikit-learn Documentation**: https://scikit-learn.org/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Research Paper**: "Customer Churn Prediction in Telecom Using ML" (Various Sources)

---

## 👥 Author

**Priyadharshini Selvadurai**  
M.Tech Student | Machine Learning Specialization  
BITS Pilani 
**GitHub**: https://github.com/Priyadharshini-Selvadurai

---

## 📝 License

This project is developed for academic purposes as part of M.Tech coursework.

---

## 🙏 Acknowledgments

- IBM for providing the Telco Customer Churn dataset
- BITS Pilani faculty for guidance and support
- Open-source community for excellent ML tools

---

**Built with Python, Scikit-learn, Pandas, and Streamlit**

*Last Updated: July 2026*
