# 🤖 Machine Learning — Complete 60-Day Roadmap
### From Absolute Beginner to Job-Ready ML Practitioner

> **Author:** Ashok | **Duration:** 60 Days | **Daily Commitment:** 2–3 hours/day

---

## 📊 Roadmap Overview

```
PHASE 1          PHASE 2          PHASE 3          PHASE 4          PHASE 5
Days 01–10       Days 11–25       Days 26–40       Days 41–52       Days 53–60
────────────     ────────────     ────────────     ────────────     ────────────
Foundations  →   Supervised   →   Unsupervised →   Advanced ML  →   Projects &
& Math           Learning         + Feature Eng.   & Pipelines       Portfolio
```

---

## 🗓️ Phase-by-Phase Plan

| Phase | Days | Topic | Goal |
|---|---|---|---|
| 🔢 **Phase 1** | 01 – 10 | Math, Python & ML Foundations | Build the base |
| 🎯 **Phase 2** | 11 – 25 | Supervised Learning | Master core algorithms |
| 🔍 **Phase 3** | 26 – 40 | Unsupervised Learning + Feature Engineering | Handle real data |
| 🚀 **Phase 4** | 41 – 52 | Advanced ML + Model Deployment | Production-ready skills |
| 🏆 **Phase 5** | 53 – 60 | Capstone Projects + Portfolio | Land the job |

---

## 📅 PHASE 1 — Foundations & Math (Days 1–10)

> **Goal:** Build the mathematical and Python programming foundation that underpins every ML algorithm.

---

### 📌 Day 1–2 — Python for ML

| Day | Topics | Practice |
|---|---|---|
| **Day 1** | NumPy arrays, broadcasting, vectorisation | 20 NumPy exercises |
| **Day 2** | Pandas DataFrames, indexing, groupby, merge | Load & explore a CSV dataset |

```python
# Day 1 — NumPy essentials
import numpy as np

A = np.array([[1, 2], [3, 4]])
print(A.T)              # Transpose
print(np.dot(A, A))     # Matrix multiplication
print(A @ A)            # Same as above
print(np.linalg.inv(A)) # Inverse

# Day 2 — Pandas essentials
import pandas as pd

df = pd.read_csv('data.csv')
print(df.describe())                # Summary stats
print(df.isnull().sum())            # Missing values
print(df.groupby('category').mean()) # Group stats
```

**Resources:**
- 📖 [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- 📖 [Pandas 10 Minutes](https://pandas.pydata.org/docs/user_guide/10min.html)

---

### 📌 Day 3–4 — Linear Algebra

| Day | Topics | Why It Matters |
|---|---|---|
| **Day 3** | Vectors, dot product, matrix ops, transpose | Neural networks, PCA |
| **Day 4** | Eigenvalues, eigenvectors, SVD basics | PCA, dimensionality reduction |

**Key Concepts:**

```
Vector Dot Product:    a · b = Σ aᵢbᵢ
Matrix Multiply:       C = A @ B   (rows × cols)
Eigendecomposition:    A · v = λ · v
```

---

### 📌 Day 5 — Statistics & Probability

| Topic | Key Formula | Use in ML |
|---|---|---|
| Mean / Variance | μ = Σx/n, σ² = Σ(x−μ)²/n | Feature scaling |
| Bayes' Theorem | P(A\|B) = P(B\|A)·P(A) / P(B) | Naive Bayes |
| Normal Distribution | f(x) = (1/σ√2π) e^(−(x−μ)²/2σ²) | Assumptions in LR |
| Correlation | ρ = Cov(X,Y) / (σₓσᵧ) | Feature selection |
| Central Limit Theorem | Sample means → Normal | Confidence intervals |

---

### 📌 Day 6 — Calculus for ML

| Topic | Key Concept | Use in ML |
|---|---|---|
| Derivatives | Rate of change of a function | Gradient descent |
| Partial Derivatives | ∂f/∂x holding other vars constant | Backpropagation |
| Chain Rule | d/dx[f(g(x))] = f'(g(x))·g'(x) | Neural networks |
| Gradient | Vector of all partial derivatives | Optimisation |

```
Gradient Descent update rule:
θ = θ − α · ∇J(θ)

Where:  α  = learning rate
        ∇J = gradient of loss function
```

---

### 📌 Day 7 — Data Visualisation

| Library | Use Case | Key Plots |
|---|---|---|
| **Matplotlib** | Custom plots | Line, scatter, histogram |
| **Seaborn** | Statistical plots | Heatmap, pairplot, boxplot |
| **Plotly** | Interactive plots | Dashboard-ready charts |

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Distribution plot
sns.histplot(df['feature'], kde=True)
plt.show()

# Pairplot
sns.pairplot(df, hue='target')
plt.show()
```

---

### 📌 Day 8–9 — ML Fundamentals

| Day | Topics |
|---|---|
| **Day 8** | What is ML? Supervised vs Unsupervised vs Reinforcement |
| **Day 8** | Train/Validation/Test split, Overfitting vs Underfitting |
| **Day 9** | Loss functions, Gradient Descent (Batch, SGD, Mini-batch) |
| **Day 9** | Bias-Variance tradeoff, Regularisation (L1, L2) |

```
Bias-Variance Tradeoff:
Total Error = Bias² + Variance + Irreducible Noise

High Bias   → Underfitting (model too simple)
High Variance → Overfitting (model too complex)
```

**Loss Functions Summary:**

| Loss | Formula | Used In |
|---|---|---|
| MSE | (1/n) Σ(y − ŷ)² | Regression |
| MAE | (1/n) Σ\|y − ŷ\| | Robust regression |
| Binary Cross-Entropy | −[y·log(ŷ) + (1−y)·log(1−ŷ)] | Classification |
| Categorical Cross-Entropy | −Σ yᵢ·log(ŷᵢ) | Multi-class |
| Hinge Loss | max(0, 1 − y·ŷ) | SVM |

---

### 📌 Day 10 — Scikit-Learn Workflow

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Universal ML workflow in sklearn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  YourModel())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### ✅ Phase 1 Checkpoint

- [ ] Can manipulate arrays and DataFrames confidently
- [ ] Understand dot products, matrix multiplication, eigenvalues
- [ ] Know what loss functions, gradients, and bias-variance mean
- [ ] Have run a basic sklearn pipeline end-to-end

---

## 📅 PHASE 2 — Supervised Learning (Days 11–25)

> **Goal:** Master every major supervised learning algorithm — both regression and classification.

---

### 📌 Day 11–12 — Linear & Polynomial Regression

| Day | Topics |
|---|---|
| **Day 11** | Simple Linear Regression, OLS, R², RMSE, MAE |
| **Day 12** | Multiple Linear Regression, Polynomial Regression, assumptions |

```
Linear Regression:   ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
OLS Solution:        θ = (XᵀX)⁻¹ Xᵀy
```

**Regression Metrics:**

| Metric | Formula | Range | Better |
|---|---|---|---|
| R² Score | 1 − SS_res/SS_tot | 0 to 1 | Higher |
| RMSE | √(Σ(y−ŷ)²/n) | 0 → ∞ | Lower |
| MAE | Σ\|y−ŷ\|/n | 0 → ∞ | Lower |
| MAPE | (1/n)Σ\|y−ŷ\|/\|y\| × 100% | 0 → ∞ | Lower |

---

### 📌 Day 13 — Ridge, Lasso & ElasticNet

| Model | Penalty | Effect | Use When |
|---|---|---|---|
| **Ridge (L2)** | λΣθ² | Shrinks all coefficients | Many small features |
| **Lasso (L1)** | λΣ\|θ\| | Sets some to zero (sparse) | Feature selection |
| **ElasticNet** | λ₁Σ\|θ\| + λ₂Σθ² | Mix of both | Many correlated features |

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV

# Ridge with CV tuning
ridge = Ridge()
params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(ridge, params, cv=5, scoring='r2')
grid.fit(X_train, y_train)
print(f"Best alpha: {grid.best_params_}")
```

---

### 📌 Day 14–15 — Logistic Regression

| Day | Topics |
|---|---|
| **Day 14** | Sigmoid function, binary classification, decision boundary |
| **Day 15** | Multi-class: One-vs-Rest, One-vs-One, Softmax |

```
Sigmoid:     σ(z) = 1 / (1 + e⁻ᶻ)
Prediction:  ŷ = σ(θᵀx)  →  class 1 if ŷ ≥ 0.5
```

**Classification Metrics:**

| Metric | Formula | Focus |
|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Avoid false positives |
| Recall | TP/(TP+FN) | Avoid false negatives |
| F1 Score | 2·P·R/(P+R) | Balance P and R |
| ROC-AUC | Area under ROC curve | Rank-ordering |

---

### 📌 Day 16 — Decision Trees

```
                  [Feature A ≤ 3.5]
                 /                  \
         [Feature B ≤ 1.0]     [Feature C ≤ 2.5]
         /           \           /            \
      Class 0      Class 1   Class 1        Class 0
```

| Concept | Detail |
|---|---|
| **Split criterion** | Gini Impurity or Information Gain (Entropy) |
| **Gini** | 1 − Σ pᵢ² (lower = purer) |
| **Entropy** | −Σ pᵢ·log₂(pᵢ) |
| **Pruning** | max_depth, min_samples_split, min_samples_leaf |
| **Pros** | Interpretable, handles mixed types, no scaling needed |
| **Cons** | High variance, prone to overfitting |

---

### 📌 Day 17–18 — Ensemble Methods

#### Random Forest (Day 17)

```
                    Random Forest
         ┌──────────────┼──────────────┐
      Tree 1         Tree 2         Tree N
      (subset)       (subset)       (subset)
         └──────────────┼──────────────┘
                   Majority Vote / Avg
```

| Parameter | Effect |
|---|---|
| `n_estimators` | More trees = more stable (diminishing returns) |
| `max_features` | Subset of features per split (controls correlation) |
| `max_depth` | Depth of each tree |
| `bootstrap` | Whether to sample with replacement |

#### Boosting (Day 18)

| Algorithm | Key Idea | Best For |
|---|---|---|
| **AdaBoost** | Upweight misclassified samples | Binary classification |
| **Gradient Boosting** | Fit residuals sequentially | Tabular regression/classification |
| **XGBoost** | Regularised GBM with speed | Kaggle competitions |
| **LightGBM** | Leaf-wise tree growth, fast | Large datasets |
| **CatBoost** | Native categorical handling | Mixed-type data |

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
```

---

### 📌 Day 19–20 — Support Vector Machines (SVM)

| Day | Topics |
|---|---|
| **Day 19** | Hard-margin SVM, support vectors, hyperplane, margin maximisation |
| **Day 20** | Soft-margin (C), Kernel trick (RBF, Polynomial, Sigmoid) |

```
Decision boundary:   wᵀx + b = 0
Margin:              2 / ‖w‖
Objective:           Minimise ½‖w‖²  subject to  yᵢ(wᵀxᵢ + b) ≥ 1
```

| Kernel | Formula | Use Case |
|---|---|---|
| Linear | K(x,z) = xᵀz | Linearly separable data |
| RBF | K(x,z) = exp(−γ‖x−z‖²) | Non-linear, general purpose |
| Polynomial | K(x,z) = (xᵀz + c)^d | Image features |

---

### 📌 Day 21 — K-Nearest Neighbours (KNN)

| Aspect | Detail |
|---|---|
| **Idea** | Classify by majority vote of K nearest neighbours |
| **Distance** | Euclidean, Manhattan, Minkowski |
| **Choosing K** | Cross-validate; small K = high variance, large K = high bias |
| **Scaling** | Mandatory — KNN is distance-based |
| **Complexity** | O(n·d) per prediction — slow on large data |

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

errors = []
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=5).mean()
    errors.append(score)

best_k = errors.index(max(errors)) + 1
print(f"Best K: {best_k}")
```

---

### 📌 Day 22 — Naive Bayes

| Variant | Assumption | Best For |
|---|---|---|
| **GaussianNB** | Features ~ Normal distribution | Continuous features |
| **MultinomialNB** | Features are counts | Text classification |
| **BernoulliNB** | Features are binary | Spam detection |

```
Bayes Rule:  P(class | x) ∝ P(x | class) · P(class)
Naive:       assumes features are conditionally independent
```

---

### 📌 Day 23–24 — Model Evaluation & Hyperparameter Tuning

#### Cross-Validation (Day 23)

| Technique | Description | Best For |
|---|---|---|
| **K-Fold CV** | Split into K folds, rotate test set | General purpose |
| **Stratified K-Fold** | Preserves class ratio in each fold | Imbalanced classes |
| **Leave-One-Out** | K = n (every point is test once) | Very small datasets |
| **Time-Series Split** | No future data leaks into training | Time-series data |

#### Hyperparameter Tuning (Day 24)

| Method | Strategy | Speed | Best For |
|---|---|---|---|
| **GridSearchCV** | Exhaustive search | Slow | Small param space |
| **RandomizedSearchCV** | Random sampling | Fast | Large param space |
| **Optuna / Bayesian** | Smart sequential search | Fastest | Complex models |

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

params = {
    'n_estimators': randint(100, 500),
    'max_depth':    randint(3, 15),
    'learning_rate': uniform(0.01, 0.3)
}

search = RandomizedSearchCV(model, params, n_iter=50,
                             cv=5, scoring='f1', random_state=42)
search.fit(X_train, y_train)
print(search.best_params_)
```

---

### 📌 Day 25 — Handling Imbalanced Datasets

| Technique | Method | Library |
|---|---|---|
| **Oversampling** | SMOTE — synthetic minority samples | `imbalanced-learn` |
| **Undersampling** | Remove majority class samples | `imbalanced-learn` |
| **Class Weights** | `class_weight='balanced'` in sklearn | Built-in |
| **Threshold Tuning** | Adjust decision threshold (not 0.5) | Manual / sklearn |
| **Evaluation** | Always use F1, Precision-Recall AUC | Never accuracy alone |

### ✅ Phase 2 Checkpoint

- [ ] Can implement all algorithms from scratch conceptually
- [ ] Understand when to use each algorithm
- [ ] Can tune hyperparameters using GridSearch / RandomSearch
- [ ] Know all classification & regression evaluation metrics

---

## 📅 PHASE 3 — Unsupervised Learning + Feature Engineering (Days 26–40)

> **Goal:** Handle real-world messy data and discover hidden structure without labels.

---

### 📌 Day 26–28 — Clustering Algorithms

| Day | Algorithm | Key Concept |
|---|---|---|
| **Day 26** | K-Means | Centroid-based, minimise WCSS |
| **Day 27** | DBSCAN | Density-based, handles noise & arbitrary shapes |
| **Day 28** | Hierarchical Clustering | Dendrogram, no K needed upfront |

#### K-Means Quick Reference

```
Objective:     J = Σₖ Σᵢ ‖xᵢ − μₖ‖²
Choose K:      Elbow method (WCSS) + Silhouette score
Init:          K-Means++ (sklearn default)
Metrics:       Silhouette, Calinski-Harabasz, Davies-Bouldin
```

#### DBSCAN Parameters

| Parameter | Effect |
|---|---|
| `eps` | Neighbourhood radius — smaller = tighter clusters |
| `min_samples` | Min points to form a core point |
| Points labeled `-1` | Noise / outliers |

#### Hierarchical Clustering

| Linkage | Merges clusters by... |
|---|---|
| Single | Min distance between any two points |
| Complete | Max distance between any two points |
| Average | Average distance between all point pairs |
| Ward | Minimises within-cluster variance (most common) |

---

### 📌 Day 29–30 — Dimensionality Reduction

| Day | Algorithm | Use Case |
|---|---|---|
| **Day 29** | PCA (Principal Component Analysis) | Linear reduction, visualisation |
| **Day 30** | t-SNE, UMAP | Non-linear, high-dim visualisation |

#### PCA Explained

```
Steps:
1. Standardise the data
2. Compute covariance matrix: C = XᵀX / (n−1)
3. Compute eigenvectors & eigenvalues of C
4. Sort by eigenvalue (variance explained)
5. Project data onto top K eigenvectors

Explained Variance Ratio: how much variance each PC captures
Cumulative Variance:      choose K where sum ≥ 95%
```

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X_scaled)
print(f"Original: {X.shape[1]} → Reduced: {X_reduced.shape[1]} features")
print(pca.explained_variance_ratio_)
```

| Method | Type | Preserves | Best For |
|---|---|---|---|
| **PCA** | Linear | Global variance structure | Preprocessing, speed |
| **t-SNE** | Non-linear | Local neighbourhood | 2D/3D visualisation |
| **UMAP** | Non-linear | Both local & global | Faster than t-SNE |

---

### 📌 Day 31–35 — Feature Engineering

> This is where most ML competitions are won or lost.

#### Day 31 — Handling Missing Values

| Strategy | When to Use | Code |
|---|---|---|
| **Drop rows** | < 5% missing | `df.dropna()` |
| **Mean/Median fill** | Numerical, MAR | `SimpleImputer(strategy='median')` |
| **Mode fill** | Categorical | `SimpleImputer(strategy='most_frequent')` |
| **KNN Imputer** | Correlated features | `KNNImputer(n_neighbors=5)` |
| **Iterative Imputer** | Complex patterns | `IterativeImputer()` |

#### Day 32 — Encoding Categorical Variables

| Technique | When | Code |
|---|---|---|
| **Label Encoding** | Ordinal categories | `LabelEncoder()` |
| **One-Hot Encoding** | Nominal, low cardinality | `pd.get_dummies()` |
| **Target Encoding** | High cardinality | `category_encoders.TargetEncoder()` |
| **Frequency Encoding** | High cardinality | `df['col'].map(df['col'].value_counts())` |

#### Day 33 — Feature Scaling

| Scaler | Formula | Use When |
|---|---|---|
| **StandardScaler** | (x − μ) / σ | Normal-ish data, SVM, LR |
| **MinMaxScaler** | (x − min)/(max − min) | Neural networks, KNN |
| **RobustScaler** | (x − median) / IQR | Data with outliers |
| **Log Transform** | log(x + 1) | Right-skewed distributions |

#### Day 34 — Feature Creation & Selection

**Creation:**
```python
# Interaction features
df['area'] = df['length'] * df['width']

# Ratio features
df['price_per_sqft'] = df['price'] / df['sqft']

# Date features
df['day_of_week'] = df['date'].dt.dayofweek
df['month']       = df['date'].dt.month
df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
```

**Selection Methods:**

| Method | Type | Code |
|---|---|---|
| Correlation filter | Filter | `df.corr()` |
| SelectKBest | Filter | `SelectKBest(f_classif, k=10)` |
| Recursive Feature Elimination | Wrapper | `RFE(estimator, n_features_to_select=10)` |
| Feature Importance | Embedded | `model.feature_importances_` |
| L1 Regularisation | Embedded | `SelectFromModel(Lasso())` |

#### Day 35 — Outlier Detection & Treatment

| Method | Approach | Code |
|---|---|---|
| **IQR Method** | Remove points outside Q1−1.5×IQR / Q3+1.5×IQR | Manual |
| **Z-Score** | Remove points where \|z\| > 3 | `scipy.stats.zscore` |
| **Isolation Forest** | Anomaly score via random trees | `IsolationForest()` |
| **Local Outlier Factor** | Density-based anomaly score | `LocalOutlierFactor()` |

---

### 📌 Day 36–38 — Association Rules & Advanced Unsupervised

| Day | Topic | Key Algorithm |
|---|---|---|
| **Day 36** | Market Basket Analysis | Apriori, FP-Growth |
| **Day 37** | Anomaly Detection | Isolation Forest, Autoencoder |
| **Day 38** | Gaussian Mixture Models | Soft clustering, EM algorithm |

#### Association Rules Metrics

| Metric | Meaning | Formula |
|---|---|---|
| **Support** | How often itemset appears | P(A ∩ B) |
| **Confidence** | How often rule is correct | P(B\|A) |
| **Lift** | Strength above chance | Conf / P(B) |

---

### 📌 Day 39–40 — Pipelines & Preprocessing Automation

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define column types
num_cols = ['age', 'salary', 'experience']
cat_cols = ['department', 'city']

# Numerical pipeline
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

# Categorical pipeline
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# Full pipeline with model
full_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model',        XGBClassifier())
])

full_pipe.fit(X_train, y_train)
```

### ✅ Phase 3 Checkpoint

- [ ] Can cluster data and choose the right algorithm for the task
- [ ] Understand PCA and when to use t-SNE vs UMAP
- [ ] Can engineer features from raw data (dates, text, categoricals)
- [ ] Can build a clean sklearn Pipeline with ColumnTransformer

---

## 📅 PHASE 4 — Advanced ML & Deployment (Days 41–52)

> **Goal:** Learn cutting-edge techniques and build production-ready skills.

---

### 📌 Day 41–44 — Neural Networks & Deep Learning Basics

| Day | Topics |
|---|---|
| **Day 41** | Perceptron, activation functions, forward pass |
| **Day 42** | Backpropagation, optimisers (SGD, Adam, RMSProp) |
| **Day 43** | Build ANN with Keras/TensorFlow for classification & regression |
| **Day 44** | Dropout, Batch Normalisation, Early Stopping |

**Activation Functions:**

| Function | Formula | Use |
|---|---|---|
| ReLU | max(0, x) | Hidden layers (default) |
| Sigmoid | 1/(1+e⁻ˣ) | Binary output |
| Softmax | eˣᵢ/Σeˣⱼ | Multi-class output |
| Tanh | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | RNNs, hidden layers |
| Leaky ReLU | max(0.01x, x) | Dying ReLU problem |

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(n_features,)),
    keras.layers.Dropout(0.3),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32,
          validation_split=0.2,
          callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
```

---

### 📌 Day 45–46 — Natural Language Processing (NLP) Basics

| Day | Topics |
|---|---|
| **Day 45** | Text preprocessing: tokenisation, stopwords, stemming, lemmatisation |
| **Day 45** | Bag of Words, TF-IDF, n-grams |
| **Day 46** | Word embeddings: Word2Vec, GloVe |
| **Day 46** | Sentiment analysis, text classification pipeline |

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

text_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('clf',   LogisticRegression(max_iter=1000))
])

text_pipe.fit(X_train_text, y_train)
print(f"Test Accuracy: {text_pipe.score(X_test_text, y_test):.4f}")
```

---

### 📌 Day 47 — Time Series Forecasting

| Concept | Detail |
|---|---|
| **Stationarity** | Constant mean/variance over time (ADF test) |
| **ACF / PACF** | Identify AR and MA orders |
| **ARIMA** | Auto-Regressive Integrated Moving Average |
| **SARIMA** | ARIMA + seasonal component |
| **Prophet** | Facebook's time-series library |
| **ML for TS** | Lag features → XGBoost / LightGBM |

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train_series, order=(2, 1, 2))
fitted = model.fit()
forecast = fitted.forecast(steps=30)
```

---

### 📌 Day 48 — Recommendation Systems

| Type | Approach | Example |
|---|---|---|
| **Content-Based** | Item features similarity | "Because you watched X" |
| **Collaborative Filtering** | User-item matrix factorisation | "Users like you watched Y" |
| **Hybrid** | Content + Collaborative | Netflix, Spotify |

---

### 📌 Day 49–50 — ML Experiment Tracking & MLOps Basics

| Tool | Purpose |
|---|---|
| **MLflow** | Track experiments, log metrics, register models |
| **Weights & Biases** | Visual experiment tracking |
| **DVC** | Data version control |
| **Great Expectations** | Data validation & testing |

```python
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 6)

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

---

### 📌 Day 51–52 — Model Deployment

| Method | Tool | Use Case |
|---|---|---|
| **REST API** | FastAPI + Uvicorn | Production APIs |
| **Containerisation** | Docker | Portable deployment |
| **Cloud Deployment** | AWS SageMaker, GCP AI Platform | Scalable serving |
| **Serverless** | AWS Lambda | Lightweight inference |

```python
# FastAPI deployment example
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(features: list[float]):
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0].max()
    return {"prediction": int(prediction), "confidence": float(probability)}
```

### ✅ Phase 4 Checkpoint

- [ ] Can build and train a neural network with Keras
- [ ] Can preprocess and classify text data
- [ ] Can log and track ML experiments with MLflow
- [ ] Can deploy a model as a REST API with FastAPI

---

## 📅 PHASE 5 — Capstone Projects & Portfolio (Days 53–60)

> **Goal:** Build 3 real-world projects and create a portfolio that gets you hired.

---

### 📌 Day 53–55 — Project 1: Structured Data (Tabular)

**🏠 House Price Prediction (Regression)**

```
Dataset:    Kaggle House Prices (Ames Housing)
Goal:       Predict sale price from 80 features
Techniques: Feature engineering, XGBoost, Stacking
Metric:     RMSLE (Root Mean Squared Log Error)

Steps:
1. EDA — distributions, correlations, outliers
2. Feature engineering — year diffs, area ratios
3. Encode categoricals — target encoding for high-cardinality
4. Model — XGBoost + LightGBM + Ridge stacking
5. Tune — Optuna / RandomizedSearchCV
6. Evaluate — CV RMSLE, feature importance plot
7. Submit — Generate predictions, document findings
```

---

### 📌 Day 56–57 — Project 2: Classification

**💳 Credit Card Fraud Detection (Imbalanced Classification)**

```
Dataset:    Kaggle Credit Card Fraud (284K transactions, 0.17% fraud)
Goal:       Detect fraudulent transactions
Techniques: SMOTE, class weights, threshold tuning
Metric:     Precision-Recall AUC, F1

Steps:
1. EDA — severe class imbalance analysis
2. Resample — SMOTE + Tomek Links
3. Model — Isolation Forest + XGBoost + Logistic Regression
4. Tune threshold — maximise F1 / Recall tradeoff
5. Evaluate — Confusion matrix, PR-AUC curve
6. Deploy — FastAPI endpoint with risk scoring
```

---

### 📌 Day 58–59 — Project 3: NLP or Clustering

**Choose One:**

```
Option A — Sentiment Analysis
   Dataset:    IMDB / Amazon Reviews
   Goal:       Classify reviews as positive/negative
   Techniques: TF-IDF + LR, Word2Vec + LSTM
   Metric:     F1 Score, ROC-AUC

Option B — Customer Segmentation
   Dataset:    Mall Customers / RFM data
   Goal:       Segment customers for marketing
   Techniques: K-Means, DBSCAN, PCA visualisation
   Metric:     Silhouette Score, business interpretation
```

---

### 📌 Day 60 — Portfolio & Job Readiness

**GitHub Portfolio Checklist:**

- [ ] Each project has a clean `README.md` with problem statement, approach, results
- [ ] Notebooks are well-commented with markdown explanations
- [ ] Include visualisations (confusion matrix, feature importance, cluster plots)
- [ ] Add a `requirements.txt` or `environment.yml` to every project
- [ ] Pin your best 3–4 repos on your GitHub profile

**LinkedIn / Resume:**
- [ ] List all algorithms you can implement and explain
- [ ] Quantify results: "Achieved 94.2% F1-score on imbalanced fraud dataset"
- [ ] Link to GitHub and deployed projects

---

## 📋 Complete Day-by-Day Schedule

| Day | Phase | Topic | Deliverable |
|---|---|---|---|
| 01 | 1 | NumPy | 20 NumPy exercises |
| 02 | 1 | Pandas | EDA on a real dataset |
| 03 | 1 | Linear Algebra | Matrix ops notebook |
| 04 | 1 | Eigenvalues & SVD | PCA from scratch |
| 05 | 1 | Statistics & Probability | Stats summary sheet |
| 06 | 1 | Calculus & Gradient Descent | GD from scratch |
| 07 | 1 | Visualisation | Matplotlib + Seaborn plots |
| 08 | 1 | ML Fundamentals | Bias-variance notebook |
| 09 | 1 | Loss Functions & Optimisation | Loss comparison |
| 10 | 1 | Sklearn Workflow | First end-to-end pipeline |
| 11 | 2 | Linear Regression | House price mini-project |
| 12 | 2 | Polynomial & Multiple Regression | Non-linear fitting |
| 13 | 2 | Ridge, Lasso, ElasticNet | Regularisation comparison |
| 14 | 2 | Logistic Regression | Binary classifier |
| 15 | 2 | Multi-class Classification | Iris / Wine dataset |
| 16 | 2 | Decision Trees | Tree visualisation |
| 17 | 2 | Random Forest | Feature importance analysis |
| 18 | 2 | XGBoost & Boosting | Titanic survival |
| 19 | 2 | SVM — Linear | Hard margin SVM |
| 20 | 2 | SVM — Kernel | RBF kernel non-linear |
| 21 | 2 | KNN | Optimal K search |
| 22 | 2 | Naive Bayes | Spam classifier |
| 23 | 2 | Cross-Validation | Stratified K-Fold |
| 24 | 2 | Hyperparameter Tuning | Optuna / RandomizedSearch |
| 25 | 2 | Imbalanced Data | SMOTE + threshold tuning |
| 26 | 3 | K-Means | Elbow + Silhouette |
| 27 | 3 | DBSCAN | Cluster arbitrary shapes |
| 28 | 3 | Hierarchical Clustering | Dendrogram |
| 29 | 3 | PCA | Dimensionality reduction |
| 30 | 3 | t-SNE & UMAP | High-dim visualisation |
| 31 | 3 | Missing Value Imputation | KNN Imputer |
| 32 | 3 | Categorical Encoding | Target encoding |
| 33 | 3 | Feature Scaling | Scaler comparison |
| 34 | 3 | Feature Selection | RFE + SHAP |
| 35 | 3 | Outlier Detection | Isolation Forest |
| 36 | 3 | Association Rules | Market basket |
| 37 | 3 | Anomaly Detection | Credit card anomalies |
| 38 | 3 | Gaussian Mixture Models | Soft clustering |
| 39 | 3 | Sklearn Pipelines | Full pipeline |
| 40 | 3 | ColumnTransformer | Automated preprocessing |
| 41 | 4 | Perceptron & ANN | Forward pass |
| 42 | 4 | Backpropagation | Manual BP |
| 43 | 4 | Keras ANN | Classification + Regression |
| 44 | 4 | Regularisation in DL | Dropout + BN |
| 45 | 4 | NLP Basics | TF-IDF pipeline |
| 46 | 4 | Word Embeddings | Word2Vec |
| 47 | 4 | Time Series | ARIMA + Prophet |
| 48 | 4 | Recommender Systems | Collaborative filtering |
| 49 | 4 | MLflow Tracking | Log experiments |
| 50 | 4 | Model Versioning | MLflow Model Registry |
| 51 | 4 | FastAPI Deployment | REST API |
| 52 | 4 | Docker | Containerise model |
| 53 | 5 | Project 1 — EDA | House Prices EDA |
| 54 | 5 | Project 1 — Model | XGBoost + Stacking |
| 55 | 5 | Project 1 — Polish | README + Notebook |
| 56 | 5 | Project 2 — EDA | Fraud detection EDA |
| 57 | 5 | Project 2 — Model | SMOTE + Threshold |
| 58 | 5 | Project 3 — Build | NLP or Clustering |
| 59 | 5 | Project 3 — Deploy | FastAPI endpoint |
| 60 | 5 | Portfolio Day | GitHub + LinkedIn |

---

## 🛠️ Tools & Libraries Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Core ML** | Scikit-learn, XGBoost, LightGBM, CatBoost |
| **Deep Learning** | TensorFlow / Keras, PyTorch (optional) |
| **Data** | NumPy, Pandas, SciPy |
| **Visualisation** | Matplotlib, Seaborn, Plotly |
| **NLP** | NLTK, SpaCy, HuggingFace Transformers |
| **Time Series** | Statsmodels, Prophet, sktime |
| **Feature Eng.** | Feature-engine, Category Encoders |
| **Imbalanced** | imbalanced-learn |
| **MLOps** | MLflow, DVC, Weights & Biases |
| **Deployment** | FastAPI, Docker, Streamlit |
| **Notebooks** | Jupyter Lab, Google Colab |

---

## 📚 Learning Resources

| Resource | Type | Best For |
|---|---|---|
| [Hands-On ML (Géron)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) | Book | Comprehensive theory + practice |
| [Kaggle Learn](https://www.kaggle.com/learn) | Free Course | Practical ML fast |
| [StatQuest (YouTube)](https://www.youtube.com/@statquest) | Video | Intuitive algorithm explanations |
| [fast.ai](https://www.fast.ai/) | Free Course | Deep learning top-down |
| [ML Mastery](https://machinelearningmastery.com/) | Blog | Code-first tutorials |
| [Papers With Code](https://paperswithcode.com/) | Research | SOTA methods & benchmarks |

---

## ⚡ Daily Study Template

```
📅 Day XX — [Topic Name]
────────────────────────────────────────
⏰ Time Block:    2.5 hours

📖 Theory        (45 min)
   → Read concept, formulas, intuition

💻 Implementation (60 min)
   → Code from scratch or adapt example

🧪 Experiment    (30 min)
   → Try on a real dataset (Kaggle / sklearn toy)

📝 Notes         (15 min)
   → Write key takeaways in your own words

🔁 Review        (10 min)
   → Revisit yesterday's notes briefly
────────────────────────────────────────
```

---

## 🏁 How to Run Projects

```bash
# Clone this repository
git clone https://github.com/ashok/ml-roadmap-60days.git
cd ml-roadmap-60days

# Create virtual environment
python -m venv ml_env
source ml_env/bin/activate      # Linux/Mac
ml_env\Scripts\activate.bat     # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

---

## 📁 Repository Structure

```
ml-roadmap-60days/
│
├── phase1_foundations/
│   ├── day01_numpy.ipynb
│   ├── day02_pandas.ipynb
│   └── ...
│
├── phase2_supervised/
│   ├── day11_linear_regression.ipynb
│   ├── day17_random_forest.ipynb
│   └── ...
│
├── phase3_unsupervised/
│   ├── day26_kmeans.ipynb
│   ├── day29_pca.ipynb
│   └── ...
│
├── phase4_advanced/
│   ├── day41_neural_networks.ipynb
│   ├── day51_fastapi_deployment/
│   └── ...
│
├── phase5_projects/
│   ├── project1_house_prices/
│   ├── project2_fraud_detection/
│   └── project3_nlp_or_clustering/
│
├── requirements.txt
└── README.md
```

---

<div align="center">

**🎯 You've got this! Consistency beats intensity — 2 hours every day for 60 days = 120 hours of focused ML learning.**

*Made with ❤️ by Ashok | Unsupervised Machine Learning Series*

</div>
