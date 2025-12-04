# Stock Market Prediction Using Machine Learning and Natural Language Processing

# Data Collection
![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/289a31be99b9b0e2b944013c9d98acaedfebeedc/stock%20market%20prediction/web%20scraping/screenshot%20data%20collection/stock%20price%20dataset%20from%20yahoo%20finance.png)

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/25d1f3c5e441e3a25495bf1a6c6dacdaaf8e8153/stock%20market%20prediction/web%20scraping/screenshot%20data%20collection/news%20stock%20dataset%20from%20yahoo%20finance.png)

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/cb975acbf58e2cc8848698a4f7e1fe78df306751/stock%20market%20prediction/web%20scraping/screenshot%20data%20collection/aapl%20price%20dataset.png)

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/1073adc910fd6fac6685d04ee64e8f95b30b0d21/stock%20market%20prediction/web%20scraping/screenshot%20data%20collection/aapl%20news%20dataset.png)

# Exploratory Data Analysis

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/c80c696a81cd3f140af3aa10675e0573cdbd9f10/stock%20market%20prediction/exploratory%20data%20analysis/eda%20all%20companies/AAPL1.png)

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/78513a4ce40ad2a17cc58ca58a8eee2414481648/stock%20market%20prediction/exploratory%20data%20analysis/eda%20all%20companies/AAPL2.png)

![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/7e16dcabb62e45da4494563cfa669cf51207ba46/stock%20market%20prediction/exploratory%20data%20analysis/normalized_stock_price_comparison.png)

# Data Preprocessing

# Feature Engineering

# Model Development

# Evaluation and Comparison for Regression Models

## Training LSTM
![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/31dac7a77328c0fc53f860b0d916faef892b1c0d/stock%20market%20prediction/regression%20models/lstm%20output/AAPL_price_vs_timestep_train.png)

## Testing LSTM
![image](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/ff65f7134cac7914b7780cbf03d1edc998765c96/stock%20market%20prediction/regression%20models/lstm%20output/AAPL_price_vs_timestep_test.png)

## Training SVR
![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/225e4f3014e263259ef79237f618caa3503a82e2/stock%20market%20prediction/regression%20models/svr%20output/AAPL_price_vs_timestep_train.png)

## Testing SVR
![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/a295cae0a771f3ef035002ada06ba76a2cd570a0/stock%20market%20prediction/regression%20models/svr%20output/AAPL_price_vs_timestep_test.png)

## Training MLPRegressor
![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/4131012d59141be7f817e07d65dba70cb1490b99/stock%20market%20prediction/regression%20models/mlpregressor%20output/AAPL_price_vs_timestep_train.png)

## Testing MLPRegressor
![image alt](https://github.com/amjadzkwn/Stock-Market-Prediction-Using-Machine-Learning-and-Natural-Language-Processing/blob/a532880c64800f0bbd60991c33de642710e7656e/stock%20market%20prediction/regression%20models/mlpregressor%20output/AAPL_price_vs_timestep_test.png)

# Evaluation and Comparison for Classification Models

## Classification Report for Logistic Regression
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

## Classification Report for SVM
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

## Classification Report for Random Forest
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

## Results Comparison
| Model | Logistic Regression | SVM | Random Forest |
|-------|-----------|--------|----------|
| **Accuracy**     | 0.7671      | 0.7630   | 0.6586     |

# Evaluation and Comparison for Sentiment Models

## Classification Report for FinBERT
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

## Classification Report for BERT
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

## Classification Report for VADER
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

## Results Comparison
| Model | FinBERT | BERT | VADER |
|-------|-----------|--------|----------|
| **Accuracy**     | 0.8649      | 0.8701   | 0.5289     |

# Models Integration

# Hybrid Model Output

# Impact of The Project
