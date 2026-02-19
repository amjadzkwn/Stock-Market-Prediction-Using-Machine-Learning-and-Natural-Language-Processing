# 📈 Stock Market Prediction Using Machine Learning & Natural Language Processing

## 🔍 Project Overview

This project explores the integration of machine learning and natural language processing (NLP) to predict stock market movements by combining historical stock price data with financial news sentiment. The system adopts a more holistic and data-driven approach to stock market prediction by leveraging both numerical time-series data and unstructured textual information.

The analysis is conducted across 10 major U.S.-based companies which are:
1. Apple Inc. (AAPL)
2. Advanced Micro Devices (AMD)
3. Amazon.com Inc. (AMZN)
4. Alphabet Inc. (GOOG)
5. Intel Corporation (INTC)
6. Meta Platforms Inc. (META)
7. Microsoft Corporation (MSFT)
8. Netflix Inc. (NFLX)
9. NVIDIA Corporation (NVDA)
10. Tesla Inc. (TSLA).

Apple Inc. (AAPL) is highlighted among these as a representative case study for detailed evaluation and visualization. The project evaluates multiple regression, classification, and sentiment analysis models which are ultimately integrated into a hybrid prediction framework to enhance predictive robustness and generalization.

## 📊 Data Collection

Two main data sources were used:

- Stock Price Dataset
  Historical stock price data collected for regression and classification tasks.
- Stock News Dataset
  Financial news articles collected via web scraping, later used for sentiment analysis.

These datasets enable the project to analyze both market behavior and investor sentiment.

**Yahoo Finance for Historical Stock Price Dataset:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/289a31be99b9b0e2b944013c9d98acaedfebeedc/stock%20market%20prediction/web%20scraping/screenshot%20data%20collection/stock%20price%20dataset%20from%20yahoo%20finance.png)

**Yahoo Finance for Financial News Dataset:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/25d1f3c5e441e3a25495bf1a6c6dacdaaf8e8153/stock%20market%20prediction/web%20scraping/screenshot%20data%20collection/news%20stock%20dataset%20from%20yahoo%20finance.png)

**AAPL Historical Stock Price Dataset:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/cb975acbf58e2cc8848698a4f7e1fe78df306751/stock%20market%20prediction/web%20scraping/screenshot%20data%20collection/aapl%20price%20dataset.png)

**AAPL Finacial News Dataset:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/1073adc910fd6fac6685d04ee64e8f95b30b0d21/stock%20market%20prediction/web%20scraping/screenshot%20data%20collection/aapl%20news%20dataset.png)

**All stock price dataset at here** [Stock Price Dataset](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/tree/bca408bd895c7484412807ff55b2ec8e9da484a8/stock%20market%20prediction/stock%20price%20dataset)

**All stock news dataset at here** [Stock News Dataset](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/tree/5bbe06656bc9306f2e5704de590b743dd2e250dd/stock%20market%20prediction/news%20stock%20dataset)

## 🔎 Exploratory Data Analysis (EDA)

EDA was conducted to:
- Understand stock price trends and volatility
- Analyze distributions of returns and technical indicators
- Identify missing values and anomalies
- Explore sentiment label distributions in news data

Insights from EDA guided feature selection and model design.

**Comprehensive Analysis AAPL I:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/c80c696a81cd3f140af3aa10675e0573cdbd9f10/stock%20market%20prediction/exploratory%20data%20analysis/eda%20all%20companies/AAPL1.png)

**Comprehensive Analysis AAPL II:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/78513a4ce40ad2a17cc58ca58a8eee2414481648/stock%20market%20prediction/exploratory%20data%20analysis/eda%20all%20companies/AAPL2.png)

**Normalized Stock Price Comparison:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/7e16dcabb62e45da4494563cfa669cf51207ba46/stock%20market%20prediction/exploratory%20data%20analysis/normalized_stock_price_comparison.png)

**Total Returned and Annualized Volatility:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/722cb3a4f89e58470468808df39dc3888d891bd2/stock%20market%20prediction/exploratory%20data%20analysis/total_returns_and_annualized_volatility.png)

**Risk-Return Profile:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/76c856bbe777153fcdeca6cdc8c7718f5aba024d/stock%20market%20prediction/exploratory%20data%20analysis/risk-return_profile.png)

## 🛠️ Data Preprocessing & Feature Engineering

Key preprocessing steps include:
- Handling missing values and outliers
- Normalization and scaling for numerical features
- Tokenization and text cleaning for news articles
- Feature extraction using technical indicators and sentiment scores

These steps ensure the data is suitable for both traditional ML models and deep learning architectures.

## 🤖 Model Development

### Regression Models

Used to predict future stock prices:
- LSTM (Long Short-Term Memory) – captures temporal dependencies in time-series data
- SVR (Support Vector Regression)
- MLP Regressor

Each model was trained and evaluated separately to compare predictive performance.

**Training LSTM:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/31dac7a77328c0fc53f860b0d916faef892b1c0d/stock%20market%20prediction/regression%20models/lstm%20output/AAPL_price_vs_timestep_train.png)

**Testing LSTM:**

![image](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/ff65f7134cac7914b7780cbf03d1edc998765c96/stock%20market%20prediction/regression%20models/lstm%20output/AAPL_price_vs_timestep_test.png)

**Training SVR:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/225e4f3014e263259ef79237f618caa3503a82e2/stock%20market%20prediction/regression%20models/svr%20output/AAPL_price_vs_timestep_train.png)

**Testing SVR:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/a295cae0a771f3ef035002ada06ba76a2cd570a0/stock%20market%20prediction/regression%20models/svr%20output/AAPL_price_vs_timestep_test.png)

**Training MLPRegressor:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/4131012d59141be7f817e07d65dba70cb1490b99/stock%20market%20prediction/regression%20models/mlpregressor%20output/AAPL_price_vs_timestep_train.png)

**Testing MLPRegressor:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/a532880c64800f0bbd60991c33de642710e7656e/stock%20market%20prediction/regression%20models/mlpregressor%20output/AAPL_price_vs_timestep_test.png)

### Classification Models

Used to predict stock movement direction (e.g. up/down):
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

**Classification Report for Logistic Regression:**

=== TRAINING ===

Accuracy: 0.7809045226130653

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.75      | 0.77   | 0.76     | 451     |
| 1     | 0.80      | 0.79   | 0.80     | 544     |
|       |           |        |          |         |
| **Accuracy** | | | 0.78 | **995** |
| **Macro Avg** | 0.78 | 0.78 | 0.78 | 995 |
| **Weighted Avg** | 0.78 | 0.78 | 0.78 | 995 |

=== TESTING ===

Accuracy: 0.7670682730923695

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.71      | 0.79   | 0.74     | 107     |
| 1     | 0.82      | 0.75   | 0.79     | 142     |
|       |           |        |          |         |
| **Accuracy** | | | 0.77 | **249** |
| **Macro Avg** | 0.76 | 0.77 | 0.77 | 249 |
| **Weighted Avg** | 0.77 | 0.77 | 0.77 | 249 |

**Classification Report for SVM:**

=== TRAINING ===

Accuracy: 0.7809045226130653

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.77      | 0.73   | 0.75     | 451     |
| 1     | 0.79      | 0.82   | 0.80     | 544     |
|       |           |        |          |         |
| **Accuracy** | | | 0.78 | **995** |
| **Macro Avg** | 0.78 | 0.78 | 0.78 | 995 |
| **Weighted Avg** | 0.78 | 0.78 | 0.78 | 995 |

=== TESTING ===

Accuracy: 0.7630522088353414

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.73      | 0.72   | 0.72     | 107     |
| 1     | 0.79      | 0.80   | 0.79     | 142     |
|       |           |        |          |         |
| **Accuracy** | | | 0.76 | **249** |
| **Macro Avg** | 0.76 | 0.76 | 0.76 | 249 |
| **Weighted Avg** | 0.76 | 0.76 | 0.76 | 249 |

**Classification Report for Random Forest:**

=== TRAINING ===

Accuracy: 0.9296482412060302

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.92      | 0.93   | 0.92     | 451     |
| 1     | 0.94      | 0.93   | 0.94     | 544     |
|       |           |        |          |         |
| **Accuracy** | | | 0.93 | **995** |
| **Macro Avg** | 0.93 | 0.93 | 0.93 | 995 |
| **Weighted Avg** | 0.93 | 0.93 | 0.93 | 995 |

=== TESTING ===

Accuracy: 0.6586345381526104

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.58      | 0.76   | 0.66     | 107     |
| 1     | 0.76      | 0.58   | 0.66     | 142     |
|       |           |        |          |         |
| **Accuracy** | | | 0.66 | **249** |
| **Macro Avg** | 0.67 | 0.67 | 0.66 | 249 |
| **Weighted Avg** | 0.68 | 0.66 | 0.66 | 249 |

**Results Comparison:**
| Model | Logistic Regression | SVM | Random Forest |
|-------|-----------|--------|----------|
| **Accuracy**     | 0.7671      | 0.7630   | 0.6586     |

Logistic Regression and SVM demonstrated better generalization, while Random Forest showed clear overfitting.

### Sentiment Analysis Models
To analyze financial news sentiment:
- FinBERT
- BERT
- VADER

**Classification Report for FinBERT:**

Accuracy: 0.8649

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Positive     | 0.86      | 0.72   | 0.78     | 273     |
| Neutral     | 0.87      | 0.93   | 0.89     | 576     |
| Negative      | 0.85          | 0.90       | 0.87         |  121       |
|       |           |        |          |         |
| **Accuracy** | | | 0.86 | **970** |
| **Macro Avg** | 0.86 | 0.85 | 0.85 | 970 |
| **Weighted Avg** | 0.86 | 0.86 | 0.86 | 970 |

**Classification Report for BERT:**

Accuracy: 0.8701

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Positive     | 0.80      | 0.81   | 0.81     | 273     |
| Neutral     | 0.91      | 0.89   | 0.90     | 576     |
| Negative      | 0.84          | 0.93       | 0.88         |  121       |
|       |           |        |          |         |
| **Accuracy** | | | 0.87 | **970** |
| **Macro Avg** | 0.85 | 0.87 | 0.86 | 970 |
| **Weighted Avg** | 0.87 | 0.87 | 0.87 | 970 |

**Classification Report for VADER:**

Accuracy: 0.5289

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Positive     | 0.38      | 0.68   | 0.49     | 273     |
| Neutral     | 0.74      | 0.50   | 0.60     | 576     |
| Negative      | 0.38          | 0.31       | 0.35         |  121       |
|       |           |        |          |         |
| **Accuracy** | | | 0.53 | **970** |
| **Macro Avg** | 0.50 | 0.50 | 0.48 | 970 |
| **Weighted Avg** | 0.60 | 0.53 | 0.54 | 970 |

**Results Comparison:**
| Model | FinBERT | BERT | VADER |
|-------|-----------|--------|----------|
| **Accuracy**     | 0.8649      | 0.8701   | 0.5289     |

Transformer-based models significantly outperformed rule-based VADER, showing their effectiveness in understanding financial text.

## 🔗 Hybrid Model Integration

A hybrid prediction model was developed by combining multiple models:

| Model | Weight |
|-------|-----------|
| **LSTM** | 60% |
| **Logistic Regression** | 30% |
| **FinBERT** | 10% |

This weighted approach integrates:
- Price trend learning (LSTM)
- Directional prediction (Logistic Regression)
- Market sentiment (FinBERT)

to produce a more robust final prediction.

## 📊 Visualization & Dashboard

An interactive dashboard was developed to visualize:
- Stock price trends
- Prediction outputs
- Portfolio action distribution
- Stock scores distribution

A total of 10 dashboards were created to provide comprehensive insights for decision-making.

**AAPL Dashboard:**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/3d0ee4eaf588f4c10654866ea344355362c4440f/stock%20market%20prediction/hybrid%20model/dashboard/AAPL_compact_dashboard.png)

**All 10 dashboard at here** [Stock Dashboard](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/tree/9ee6b49f7fa782f5df655fd528e40e5e12174a08/stock%20market%20prediction/hybrid%20model/dashboard)

**Portfolio Action Distribution and Stock Scores Distribution**

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/42163a7b8dc61e7ddcdad11a9c811800fc34ffb0/stock%20market%20prediction/hybrid%20model/portfolio_overview.png)

## 🚀 Impact of the Project

This project demonstrates how AI-driven analytics can enhance stock market prediction by:
- Combining structured and unstructured data
- Reducing reliance on price data alone
- Providing explainable and visual insights
- Supporting data-driven investment decisions

The study show the practical application of machine learning, deep learning, NLP, and data visualization in real-world financial analytics.
