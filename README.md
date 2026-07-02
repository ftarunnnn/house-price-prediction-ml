# 🏠 House Price Prediction

> An end-to-end Machine Learning project that predicts residential house prices using a Streamlit web application — from raw data generation to a deployed, interactive web app.

---

## 🎯 Project Overview

This project follows a complete real-world ML workflow:

1. **Data Generation** – Synthetic dataset of 500 houses with realistic features
2. **Data Cleaning** – Deduplication, missing value imputation, type conversion
3. **EDA** – Visual exploration of distributions, correlations, and feature relationships
4. **Feature Engineering** – One-hot encoding, scaling, derived features
5. **Model Training** – Multiple regression algorithms compared head-to-head
6. **Model Selection** – Best-performing model saved for inference
7. **Web App** – Interactive Streamlit interface for real-time predictions
8. **Deployment** – Hosted on Streamlit Community Cloud

---

## 🚀 Live Demo

> [**Open the App →**](https://your-app-url.streamlit.app)
> *(Replace with your deployed URL after deployment)*

---

## 🛠️ Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python 3.10+ |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Model Saving | Joblib |
| Web Framework | Streamlit |
| Version Control | Git, GitHub |

---

## 📂 Project Structure

```
House_Price_Prediction/
│
├── app.py                    # Streamlit web application
├── generate_dataset.py       # Synthetic dataset generator
├── requirements.txt          # Python dependencies
├── .gitignore                # Git exclusions
│
├── data/
│   └── house_prices.csv      # Dataset (500+ rows)
│
├── models/
│   ├── house_model.pkl       # Trained best model (Linear Regression)
│   ├── scaler.pkl            # Fitted StandardScaler
│   └── feature_columns.json  # Feature column names for inference
│
├── notebook/
│   └── EDA.ipynb             # Exploratory Data Analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Data cleaning pipeline
│   ├── feature_engineering.py# Encoding and scaling
│   ├── split_data.py         # Train/test split utilities
│   ├── train.py              # Model training & evaluation
│   ├── predict.py            # Inference helpers
│   └── validate.py           # Input validation
│
└── assets/
    ├── logo.png
    └── house.png
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/House_Price_Prediction.git
cd House_Price_Prediction
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Generate the Dataset

```bash
python generate_dataset.py
```

This creates `data/house_prices.csv` with 500+ synthetic house records including:
- Area (sq ft), Bedrooms, Bathrooms, Floors, Garage
- House Age, Condition, Location (Urban / Semi Urban / Rural)
- Realistic price formula with added noise

---

## 🤖 Train the Model

```bash
python -m src.train
```

This will:
1. Load and clean the dataset
2. Apply feature engineering
3. Train **5 regression models**: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost
4. Print a comparison table (MAE, RMSE, R²)
5. Save the **best model** to `models/house_model.pkl`
6. Save the **scaler** to `models/scaler.pkl`

### Model Performance Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~1,20,000 | ~1,60,000 | **0.97** |
| Gradient Boosting | ~1,35,000 | ~1,80,000 | 0.96 |
| Random Forest | ~1,50,000 | ~1,95,000 | 0.95 |
| Decision Tree | ~2,10,000 | ~2,80,000 | 0.92 |
| XGBoost | ~1,40,000 | ~1,85,000 | 0.96 |

> ✅ **Best Model: Linear Regression** with R² ≈ 0.97

---

## 🌐 Run the App Locally

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Streamlit Workflow

```
User Opens App
      │
      ▼
Enter House Details (Sidebar)
      │
      ▼
Click "Predict Price"
      │
      ▼
Input Validation
      │
      ▼
Load Trained Model (cached)
      │
      ▼
Preprocess Input
      │
      ▼
Predict Price
      │
      ▼
Display Estimated Price (₹)
```

---

## ☁️ Deployment

### Option 1: Streamlit Community Cloud (Free & Recommended)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New App**
4. Select your repository and branch
5. Set **Main file path** to `app.py`
6. Click **Deploy** → get a public URL instantly

### Option 2: Render

1. Create a new **Web Service** on [render.com](https://render.com)
2. Connect your GitHub repository
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

### Option 3: Railway

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Add environment variable: `PORT=8501`
3. Set start command same as Render above

### Option 4: Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Streamlit** as the SDK
3. Push your repository — HF auto-detects `app.py` and `requirements.txt`

---

## ✅ Final Deliverables

| Deliverable | Status |
|---|---|
| Cleaned house price dataset | ✅ |
| EDA notebook | ✅ |
| Feature engineering pipeline | ✅ |
| Multiple trained regression models | ✅ |
| Best model saved as `house_model.pkl` | ✅ |
| Preprocessing objects saved for inference | ✅ |
| Interactive Streamlit web application | ✅ |
| `requirements.txt` | ✅ |
| `README.md` | ✅ |
| GitHub repository ready for deployment | ✅ |

---

## 👤 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-profile)

---

## 📄 License

This project is licensed under the MIT License.
