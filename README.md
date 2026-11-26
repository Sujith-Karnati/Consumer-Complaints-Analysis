# 📊 SupportIQ – Customer Complaint Classification Using Machine Learning

An NLP-powered machine learning project that automatically classifies consumer financial complaints into different product categories using Natural Language Processing (NLP) and Machine Learning techniques.

The project uses the **Consumer Financial Protection Bureau (CFPB) Consumer Complaint Dataset** to build and evaluate text classification models that can understand customer complaints and predict their corresponding financial product categories.

## 🚀 Project Overview

Financial institutions receive thousands of consumer complaints related to credit reporting, debt collection, mortgages, credit cards, banking services, and other financial products.

Manually categorizing these complaints is time-consuming and difficult to scale.

**SupportIQ** aims to automate this process using NLP and machine learning. It analyzes customer complaint narratives and predicts the relevant financial product category.

The project focuses on text preprocessing, feature extraction, model training, class imbalance handling, and performance evaluation.

## ✨ Key Features

- 📝 **Text Preprocessing:** Cleans and prepares customer complaint narratives.
- 🔤 **Text Vectorization:** Uses Bag of Words and TF-IDF techniques.
- 🤖 **Machine Learning Models:** Implements multiple classification algorithms.
- ⚖️ **Class Imbalance Handling:** Uses balanced class weights to improve minority-class performance.
- 📊 **Model Evaluation:** Evaluates models using Accuracy, Precision, Recall, and F1-score.
- 🎯 **Hyperparameter Tuning:** Optimizes model parameters for improved performance.
- 📈 **Comparative Analysis:** Compares multiple machine learning algorithms.

## 📂 Dataset

**Dataset:** Consumer Financial Protection Bureau (CFPB) Consumer Complaints Dataset.

**Source:** Kaggle – Consumer Complaints Dataset

The dataset contains consumer complaint narratives and their corresponding financial product categories.

### Dataset Information

| Description | Details |
|---|---:|
| Original Records | 3,093,136 |
| Records with Narratives | 1,111,330 |
| Final Cleaned Records | 970,109 |
| Product Categories | 18 |
| Duplicate Narratives Identified | 140,519 |

### Major Product Categories

- Credit Reporting
- Debt Collection
- Mortgage
- Credit Card / Prepaid Card
- Checking / Savings Account
- Vehicle Loans
- Student Loans
- Money Transfers
- Other Financial Products

*Note: The final dataset contains 18 product categories.*

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Operations |
| NLTK | Text Preprocessing |
| Scikit-learn | Machine Learning |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| TF-IDF | Text Feature Extraction |
| Bag of Words | Text Vectorization |
| Jupyter Notebook / Google Colab | Development Environment |

## 🔄 Project Workflow

### 1. Data Collection
- Obtained the CFPB consumer complaint dataset.
- Selected complaint narratives and product labels.
- Examined missing values and dataset structure.

### 2. Exploratory Data Analysis (EDA)
- Analyzed product category distribution.
- Examined missing complaint narratives.
- Identified duplicate complaint narratives.
- Analyzed text length and word distribution.

### 3. Data Cleaning and Preprocessing

Applied the following NLP preprocessing techniques:

- Lowercasing
- Removing unnecessary characters
- Removing special symbols
- Tokenization
- Stopword removal
- Text normalization
- Handling duplicate records

### 4. Feature Engineering and Vectorization

Implemented two text vectorization approaches:

**Bag of Words (BoW)**
- Converts text into numerical word-frequency representations.
- Unigram features: 16,002.

**TF-IDF (Term Frequency–Inverse Document Frequency)**
- Assigns importance to words based on their frequency and distribution across documents.
- Combined TF-IDF features: 100,000.

### 5. Model Training

Implemented and evaluated the following machine learning algorithms:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Classifier (Linear SVC)
4. Balanced Linear SVC

### 6. Hyperparameter Tuning

Applied hyperparameter optimization to improve model performance and address class imbalance.

Best Balanced Linear SVC parameters:

```python
{
    'C': 0.02161894240657444,
    'class_weight': 'balanced'
}
```

## 📊 Model Performance

The following results were obtained during model evaluation:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Multinomial Naive Bayes | 79.25% | 30.61% | 77.47% |
| Logistic Regression | 85.23% | 40.79% | 84.54% |
| Linear SVC | 85.04% | 44.05% | 84.39% |
| Balanced Linear SVC | 82.95% | 44.37% | 83.43% |

### Best Performing Model

**Balanced Linear SVC**

- Accuracy: 82.95%
- Macro Precision: 48.51%
- Macro Recall: 45.82%
- Macro F1-score: 44.37%
- Weighted F1-score: 83.43%
- Training Time: 105.86 seconds

The Balanced Linear SVC achieved the highest Macro F1-score among the evaluated models, indicating better performance across minority product categories.

## 📁 Project Structure

```text
SupportIQ/
│
├── notebooks/
│   ├── 01_Data_Collection_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   └── 04_Model_Training_Evaluation.ipynb
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── features/
│   └── README.md
│
├── requirements.txt
│
└── README.md
```

*Update the folder and notebook names to match your actual repository structure.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/SupportIQ.git
```

### 2. Navigate to the Project Directory

```bash
cd SupportIQ
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Notebooks

```bash
jupyter notebook
```

Open the notebooks in sequence to explore the complete machine learning workflow.

## 📦 Requirements

```text
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
jupyter
scipy
joblib
```

## 💡 Challenges Faced

During the development of this project, several challenges were addressed:

- Handling a large dataset containing millions of records.
- Managing missing complaint narratives.
- Identifying and removing duplicate complaint texts.
- Processing large volumes of unstructured text.
- Handling imbalanced product categories.
- Optimizing TF-IDF feature extraction.
- Improving minority-class classification performance.
- Comparing different machine learning algorithms using appropriate evaluation metrics.

## 🔮 Future Improvements

- Develop a Streamlit-based interactive complaint classification application.
- Integrate deep learning models such as LSTM and GRU.
- Experiment with transformer-based NLP models such as BERT.
- Implement complaint sentiment analysis.
- Improve minority-class classification performance.
- Deploy the final model as a REST API.

## 👨‍💻 Author

**Sujith Karnati**

B.Tech – Computer Science and Engineering (AI & ML)

**Areas of Interest:** Machine Learning | Data Science | Natural Language Processing | Deep Learning

## ⭐ Support

If you find this project useful or interesting, please consider giving the repository a ⭐ on GitHub!

---

**SupportIQ – Transforming Consumer Complaints into Meaningful Insights Using Machine Learning.** 🚀
