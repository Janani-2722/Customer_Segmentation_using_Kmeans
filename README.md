# 🎯 Customer Segmentation using K-Means Clustering

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-link-here)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)

---

## 📌 Project Overview

This project applies **unsupervised machine learning** to segment credit card customers into distinct groups based on their spending behavior. By identifying 6 unique customer segments from 8,950 records, businesses can design targeted marketing strategies for each group.

---

## 🗂️ Dataset

- **Source:** Credit Card Customer Dataset
- **Records:** 8,950 customers
- **Features:** 17 behavioral features including Balance, Purchases, Cash Advance, Credit Limit, Payments, and Tenure

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | ML algorithms |
| Streamlit | Web app deployment |
| Pickle | Model serialization |

---

## 🚀 Project Workflow

```
Data Loading → EDA → Null Handling → Outlier Treatment → 
Feature Scaling → Elbow Method → K-Means Clustering → 
Cluster Profiling → Algorithm Comparison → PCA Visualization → Deployment
```

### 1. Exploratory Data Analysis (EDA)
- Distribution plots, boxplots, and correlation heatmap
- Identified missing values in `MINIMUM_PAYMENTS` and `CREDIT_LIMIT`
- Filled nulls with column mean

### 2. Outlier Treatment
- Detected right-skewed distributions across 8 features
- Applied `np.log1p()` transformation to compress outliers
- Visualized before vs after distributions for all skewed columns

### 3. Feature Scaling
- Applied `StandardScaler` after log transformation
- Ensures all features contribute equally to distance calculations

### 4. Finding Optimal K
- Used **Elbow Method** to identify optimal number of clusters
- Validated with **Silhouette Score**

### 5. K-Means Clustering
- Final model: **K-Means with 6 clusters**
- Silhouette Score: **0.204** (acceptable for high-dimensional real-world data)

### 6. Algorithm Comparison

| Algorithm | Silhouette Score |
|-----------|-----------------|
| K-Means | 0.204 |
| Agglomerative | - |
| DBSCAN | - |

> K-Means selected as final model for best balance of performance and interpretability.

### 7. PCA Visualization
- Reduced 17 features to 2 components for visualization
- PCA used **only for visualization** — clustering was performed on full feature set

---

## 🔍 Customer Segments Identified

| Cluster | Segment Name | Description |
|---------|-------------|-------------|
| 0 | Low Spenders, Low Balance | Minimal activity, low balance |
| 1 | High Cash Advance Users | Frequent cash withdrawals, high risk |
| 2 | Installment Buyers | Prefer buying in installments, steady behavior |
| 3 | Big One-Off Purchasers | Large occasional purchases |
| 4 | Inactive / Dormant Users | Very little to no account activity |
| 5 | Full Payment, High Credit | Pay in full regularly, most financially healthy |

---

## 📊 Visualizations

| Chart | Insight |
|-------|---------|
| Before vs After Log Transform | Shows outlier treatment effectiveness |
| Elbow Curve | Optimal K selection |
| Silhouette Score Comparison | Algorithm benchmarking |
| PCA Scatter Plot | 2D cluster visualization |
| Cluster Bar Chart | Customer count per segment |

---

## 🌐 Live Demo

👉 **[Try the app here](https://customersegmentationusingkmeans-frxhmsvp27cdy3m2rktiz4.streamlit.app/)**

Enter any customer's credit card details and instantly find which segment they belong to!

---

## 📁 Project Structure

```
Marketing_Project/
│
├── Marketing_ analysis/
│   ├── app.py                          # Streamlit web app
│   ├── Marketing_segmentation.ipynb    # Main notebook
│   ├── Marketing_data.csv              # Dataset
│   ├── kmeans_model.pkl                # Saved K-Means model
│   ├── scaler.pkl                      # Saved StandardScaler
│   └── requirements.txt               # Dependencies
│
└── README.md
```

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/marketing-segmentation.git

# 2. Navigate to project folder
cd marketing-segmentation/Marketing_\ analysis

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
```

---

## 💡 Key Learnings

- Real-world customer data rarely forms perfect clusters — silhouette scores below 0.3 are common and acceptable
- Log transformation is preferred over outlier removal for financial data to preserve all records
- Clustering before PCA gives more accurate segments than clustering after PCA
- K-Means chosen over DBSCAN and Agglomerative for its interpretability and speed on large datasets

---

## 👩‍💻 Author

**Janani T** — Aspiring Data Scientist  
📍 Tamil Nadu, India  
🔗 [LinkedIn](https://www.linkedin.com/in/janani-t-168b39224/) | [GitHub](https://github.com/Janani-2722/)

---

⭐ If you found this project useful, please give it a star!
