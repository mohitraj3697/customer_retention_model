# Customer Retention Model — Churn Predictor

A deep learning project that predicts whether a bank customer is likely to leave (churn) based on their profile and account activity. Built with TensorFlow/Keras and served through an interactive Streamlit web app.

## What This Project Does

Customer churn is one of the biggest problems in banking — losing a customer costs way more than retaining one. This project takes real customer data (credit score, age, account balance, number of products, etc.) and uses a trained neural network to predict the probability of that customer churning.

The end result is a clean Streamlit interface where you punch in a customer's details and instantly get a churn probability with a clear verdict — **"likely to stay"** or **"likely to churn"**.

---

## Model Architecture

The model is a feedforward neural network built with **TensorFlow/Keras**, designed to be lightweight yet effective for binary classification.

| Layer | Type | Units | Activation | Notes |
|-------|------|-------|------------|-------|
| 1 | Dense | 32 | ReLU | Input layer — takes all scaled features |
| 2 | Dropout | — | — | 20% dropout to prevent overfitting |
| 3 | Dense | 16 | ReLU | Hidden layer |
| 4 | Dropout | — | — | 20% dropout |
| 5 | Dense | 8 | ReLU | Hidden layer |
| 6 | Dense | 1 | Sigmoid | Output — churn probability (0 to 1) |

- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Early Stopping:** Patience of 15 epochs on validation loss, restores best weights automatically
- **Batch Size:** 32
- **Max Epochs:** 200 (early stopping usually kicks in well before this)

### Why This Architecture?

I went with ReLU activations instead of sigmoid in hidden layers to avoid the vanishing gradient problem. The network is intentionally kept small (32 → 16 → 8) because the dataset has only ~10K rows and 11 features — a bigger network would just overfit. Dropout layers at 20% add regularization without killing the learning capacity.

---

## Training Results

The model was trained on the **Churn_Modelling** dataset (10,000 bank customer records) with an 80/20 train-test split (stratified to maintain class balance).

### Overall Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **85.15%** |
| Epochs Trained | 93 / 200 (early stopping) |
| Validation Accuracy | ~86.6% |
| Validation Loss | ~0.327 |

### Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Stay (0)** | 0.86 | 0.96 | 0.91 | 1,593 |
| **Churn (1)** | 0.75 | 0.41 | 0.53 | 407 |
| **Weighted Avg** | 0.84 | 0.85 | 0.83 | 2,000 |

> **Note:** The recall for churn (0.41) is lower because the dataset is imbalanced — only ~20% of customers actually churned. The model is conservative in predicting churn, meaning when it does predict churn, it's right 75% of the time (precision). For production use, you'd want to tune the threshold or apply class weights to catch more churners.

---

## Project Structure

```
customer_retention_model/
├── train_model.py         # Data preprocessing, model training, saves artifacts
├── app.py                 # Streamlit frontend for predictions
├── churn_model.keras      # Trained Keras model (generated after training)
├── scaler.pkl             # Fitted StandardScaler (generated after training)
├── feature_columns.pkl    # Feature column order (generated after training)
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker config for containerized deployment
└── Readme.md              # You're reading this
```

---

## Data Preprocessing

Here's what happens to the raw data before it hits the model:

1. **Dropped columns:** `RowNumber`, `CustomerId`, `Surname` — these are identifiers, not features
2. **One-hot encoding:** `Geography` (France/Germany/Spain) and `Gender` (Male/Female) are encoded with `drop_first=True` to avoid the dummy variable trap
3. **Feature scaling:** All features are standardized using `StandardScaler` (zero mean, unit variance) — this is critical for neural networks to converge properly
4. **Train/Test split:** 80/20 with `stratify=y` to maintain the class distribution

### Input Features (after preprocessing)

| Feature | Type | Description |
|---------|------|-------------|
| CreditScore | Numeric | Customer's credit score (300–900) |
| Age | Numeric | Customer's age |
| Tenure | Numeric | Years with the bank (0–10) |
| Balance | Numeric | Account balance |
| NumOfProducts | Numeric | Number of bank products used (1–4) |
| HasCrCard | Binary | Has a credit card (0/1) |
| IsActiveMember | Binary | Is an active member (0/1) |
| EstimatedSalary | Numeric | Estimated annual salary |
| Geography_Germany | Binary | One-hot: is from Germany |
| Geography_Spain | Binary | One-hot: is from Spain |
| Gender_Male | Binary | One-hot: is male |

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py --data Churn_Modelling.csv
```

This generates three files in the current directory:
- `churn_model.keras` — the trained neural network
- `scaler.pkl` — the fitted StandardScaler
- `feature_columns.pkl` — exact column order used during training

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Opens a browser tab where you can enter customer details and get an instant churn prediction. There's also a **"🎲 Random Customer"** button if you just want to quickly test with random inputs.

---

## Docker Deployment

```bash
docker build -t churn-predictor .
docker run -p 8501:8501 churn-predictor
```

Then open `http://localhost:8501` in your browser.

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repo and point it to `app.py`
4. Make sure `churn_model.keras`, `scaler.pkl`, and `feature_columns.pkl` are in the repo

---

## Tech Stack

- **Python 3.11**
- **TensorFlow/Keras** — model building and training
- **scikit-learn** — preprocessing (StandardScaler, train/test split, metrics)
- **pandas & NumPy** — data manipulation
- **Streamlit** — interactive web frontend
- **Docker** — containerized deployment

---

## What I Fixed from the Original Notebook

The original notebook had a few issues that would break things in production:

1. **Unscaled training data** — A `StandardScaler` was created but never actually applied to the training data before `model.fit()`. The model was learning on raw, unscaled features.
2. **Wrong prediction logic** — Used `argmax` on a single sigmoid output, which always returns 0. Fixed it to use a proper threshold: `(prediction > 0.5)`.
3. **Overkill architecture** — 9 sigmoid layers trained for 10,000 epochs leads to vanishing gradients and wasted compute. Replaced with a compact ReLU network + early stopping.

---

## Author

**Mohit Raj**
- GitHub: [@mohitraj3697](https://github.com/mohitraj3697)