
import os
import re
import html
import joblib
import nltk
import streamlit as st

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ComplaintIQ | Consumer Complaint Classifier",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #101a38, #20275c);
    color: #f5f7ff;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #101827;
    border-right: 1px solid #30394f;
}

section[data-testid="stSidebar"] * {
    color: #e9edfa;
}

/* Sidebar title */
.sidebar-title {
    font-size: 23px;
    font-weight: 800;
    color: #ffffff;
}

/* Main heading */
.main-heading {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(
        90deg,
        #ffffff,
        #a78bfa,
        #60a5fa
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    color: #b8c5df;
    font-size: 15px;
    margin-bottom: 25px;
}

/* Feature cards */
.feature-card {
    background: #202b49;
    border: 1px solid #354365;
    border-radius: 15px;
    padding: 22px;
    min-height: 135px;
    margin-bottom: 15px;
}

.feature-card h3 {
    color: #ffffff;
    font-size: 19px;
    margin-bottom: 10px;
}

.feature-card p {
    color: #bdc9df;
    font-size: 14px;
    line-height: 1.6;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #202b49;
    border: 1px solid #354365;
    border-radius: 13px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #bdc9df;
}

[data-testid="stMetricValue"] {
    color: #ffffff;
}

/* Text area */
.stTextArea textarea {
    background: #202b49 !important;
    color: #ffffff !important;
    border: 1px solid #485779 !important;
    border-radius: 10px !important;
    font-size: 15px;
}

.stTextArea textarea::placeholder {
    color: #aab6cf !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #9333ea, #2563eb);
    color: white !important;
    border: none !important;
    border-radius: 9px;
    font-weight: 650;
    min-height: 42px;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(90deg, #7e22ce, #1d4ed8);
    transform: translateY(-1px);
}

/* Prediction result */
.result-card {
    background: linear-gradient(120deg, #075b4c, #08745a);
    border: 1px solid #159b78;
    border-radius: 15px;
    padding: 32px 20px;
    text-align: center;
    margin-top: 18px;
}

.result-emoji {
    font-size: 45px;
}

.result-title {
    color: white;
    font-size: 24px;
    font-weight: 800;
    margin-top: 12px;
    overflow-wrap: anywhere;
}

.result-description {
    color: #d4f5e9;
    font-size: 14px;
    margin-top: 10px;
}

/* Sample buttons */
div[data-testid="stButton"] button {
    white-space: normal;
}

/* General text */
p, label {
    color: #dce5f5;
}

h1, h2, h3 {
    color: #ffffff;
}

/* Dividers */
hr {
    border-color: #354365 !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #202b49;
    border: 1px solid #354365;
    border-radius: 10px;
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "final_svc_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "tfidf_unigram.pkl"
)


# =========================================================
# 4. NLTK INITIALIZATION
# =========================================================

@st.cache_resource
def initialize_nltk():

    resources = [
        "stopwords",
        "wordnet",
        "omw-1.4",
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger_eng"
    ]

    for resource in resources:

        try:

            success = nltk.download(
                resource,
                quiet=True
            )

            if not success:
                raise RuntimeError(
                    f"Failed to download NLTK resource: {resource}"
                )

        except Exception as error:

            raise RuntimeError(
                f"NLTK initialization failed for {resource}: {error}"
            )

    return True


# =========================================================
# 5. LOAD MODEL AND VECTORIZER
# =========================================================

@st.cache_resource
def load_artifacts():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Vectorizer file not found: {VECTORIZER_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


# Initialize resources

try:

    initialize_nltk()

    model, vectorizer = load_artifacts()

    stop_words = set(
        stopwords.words("english")
    )

    # Preserve negation words
    negation_words = {
        "no",
        "not",
        "nor",
        "never",
        "neither",
        "none",
        "cannot"
    }

    stop_words = stop_words - negation_words

    lemmatizer = WordNetLemmatizer()

except Exception as error:

    st.error(
        f"Application initialization failed: {error}"
    )

    st.stop()


# =========================================================
# 6. TEXT CLEANING
# Matches the preprocessing notebook
# =========================================================

def clean_text(text):

    if text is None:
        return ""

    # Convert to string
    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(
        r'<[^>]+>',
        ' ',
        text
    )

    # Remove URLs
    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' ',
        text
    )

    # Remove email addresses
    text = re.sub(
        r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b',
        ' ',
        text
    )

    # Normalize contractions
    text = re.sub(
        r"won't",
        "will not",
        text
    )

    text = re.sub(
        r"can't",
        "can not",
        text
    )

    text = re.sub(
        r"n't\b",
        " not",
        text
    )

    text = re.sub(
        r"'re\b",
        " are",
        text
    )

    text = re.sub(
        r"'ve\b",
        " have",
        text
    )

    text = re.sub(
        r"'ll\b",
        " will",
        text
    )

    text = re.sub(
        r"'d\b",
        " would",
        text
    )

    text = re.sub(
        r"'m\b",
        " am",
        text
    )

    # Preserve decimal numbers
    text = re.sub(
        r'(?<=\d)\.(?=\d)',
        'DECIMALPOINT',
        text
    )

    # Remove special characters
    text = re.sub(
        r'[^a-z0-9\s]',
        ' ',
        text
    )

    # Restore decimal points
    text = text.replace(
        'DECIMALPOINT',
        '.'
    )

    # Remove extra whitespace
    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


# =========================================================
# 7. STOPWORD REMOVAL
# =========================================================

def remove_stopwords(text):

    words = text.split()

    filtered_words = [
        word
        for word in words
        if word not in stop_words
    ]

    return ' '.join(filtered_words)


# =========================================================
# 8. POS TAGGING
# =========================================================

def get_wordnet_pos(tag):

    if tag.startswith('J'):
        return wordnet.ADJ

    elif tag.startswith('V'):
        return wordnet.VERB

    elif tag.startswith('N'):
        return wordnet.NOUN

    elif tag.startswith('R'):
        return wordnet.ADV

    else:
        return wordnet.NOUN


# =========================================================
# 9. LEMMATIZATION
# =========================================================

def lemmatize_text(text):

    words = word_tokenize(text)

    tagged_words = pos_tag(words)

    lemmatized_words = []

    for word, tag in tagged_words:

        pos = get_wordnet_pos(tag)

        lemma = lemmatizer.lemmatize(
            word,
            pos
        )

        lemmatized_words.append(lemma)

    return ' '.join(lemmatized_words)


# =========================================================
# 10. COMPLETE PREPROCESSING PIPELINE
# =========================================================

def preprocess_complaint(text):

    # Step 1: Clean text
    cleaned_text = clean_text(text)

    # Step 2: Remove stopwords
    text_without_stopwords = remove_stopwords(
        cleaned_text
    )

    # Step 3: Lemmatize text
    lemmatized_text = lemmatize_text(
        text_without_stopwords
    )

    # Step 4: Remove masked placeholders
    lemmatized_text = re.sub(
        r'\bx{2,}\b',
        ' ',
        lemmatized_text,
        flags=re.IGNORECASE
    )

    # Step 5: Normalize spaces
    lemmatized_text = re.sub(
        r'\s+',
        ' ',
        lemmatized_text
    ).strip()

    return lemmatized_text


# =========================================================
# 11. PREDICTION FUNCTION
# =========================================================

def predict_complaint(complaint):

    processed_text = preprocess_complaint(
        complaint
    )

    if not processed_text.strip():

        raise ValueError(
            "No meaningful text found after preprocessing."
        )

    # TF-IDF transformation
    transformed_text = vectorizer.transform(
        [processed_text]
    )

    # Check feature dimensions
    if transformed_text.shape[1] != model.n_features_in_:

        raise ValueError(
            "TF-IDF feature count does not match the trained model."
        )

    # Predict category
    prediction = model.predict(
        transformed_text
    )[0]

    return prediction, processed_text


# =========================================================
# 12. SESSION STATE
# =========================================================

if "complaint_input" not in st.session_state:
    st.session_state.complaint_input = ""

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Dashboard"


def set_sample(text):

    st.session_state.complaint_input = text

    st.session_state.prediction_result = None


def clear_complaint():

    st.session_state.complaint_input = ""

    st.session_state.prediction_result = None


# =========================================================
# 13. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## 📊 ComplaintIQ"
    )

    st.caption(
        "Consumer Complaint Classification"
    )

    st.divider()

    st.markdown(
        "### 📌 Project Information"
    )

    st.markdown("""
    **Project:** Consumer Complaint Classification

    **Domain:** Machine Learning

    **Algorithm:** Linear SVC

    **Feature Extraction:** TF-IDF Unigram

    **Class Weight:** Balanced
    """)

    st.divider()

    st.markdown(
        "### ⚙️ Technologies"
    )

    st.markdown("""
    🐍 Python

    🧠 Machine Learning

    📚 Scikit-learn

    📝 Natural Language Processing

    📊 Streamlit

    🔤 NLTK

    🔢 TF-IDF
    """)

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Classify Complaint",
            "Model Performance",
            "About Project"
        ],
        key="selected_page"
    )

    st.divider()

    st.caption(
        "Developed by Sujith Karnati"
    )


# =========================================================
# 14. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-heading">'
        '📊 ComplaintIQ – Consumer Complaint Classification'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Discover the insights behind consumer financial complaints.'
        '<br>Powered by Natural Language Processing and Machine Learning.'
        '</div>',
        unsafe_allow_html=True
    )

    # Feature cards

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="feature-card">
            <h3>🧠 AI Powered</h3>
            <p>
                Machine Learning based consumer complaint
                classification using Natural Language Processing.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Fast Analysis</h3>
            <p>
                Analyze consumer complaint narratives and
                predict the relevant category instantly.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="feature-card">
            <h3>🎯 17 Categories</h3>
            <p>
                Classifies complaints into the financial
                product categories learned by the trained model.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        "### ✨ Try Sample Complaints"
    )

    sample_credit_card = (
        "I noticed unauthorized transactions on my credit "
        "card statement and the bank has not refunded my money."
    )

    sample_debt = (
        "A debt collection agency keeps calling me about "
        "a loan that I have already paid."
    )

    sample_credit_report = (
        "My credit report contains incorrect information "
        "about an account that does not belong to me."
    )

    sample_mortgage = (
        "My mortgage company has not applied my monthly "
        "payment correctly and is charging late fees."
    )

    sample_bank = (
        "My bank account was frozen without any explanation "
        "and I cannot access my money."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.button(
            "💳 Try Credit Card Complaint",
            use_container_width=True,
            on_click=set_sample,
            args=(sample_credit_card,)
        )

        st.button(
            "📄 Try Credit Report Complaint",
            use_container_width=True,
            on_click=set_sample,
            args=(sample_credit_report,)
        )

        st.button(
            "🏠 Try Mortgage Complaint",
            use_container_width=True,
            on_click=set_sample,
            args=(sample_mortgage,)
        )

    with col2:

        st.button(
            "📞 Try Debt Collection Complaint",
            use_container_width=True,
            on_click=set_sample,
            args=(sample_debt,)
        )

        st.button(
            "🏦 Try Bank Account Complaint",
            use_container_width=True,
            on_click=set_sample,
            args=(sample_bank,)
        )

    st.divider()

    st.markdown(
        "### 📈 Model Overview"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Accuracy",
            "82.945%"
        )

    with col2:

        st.metric(
            "Macro F1-Score",
            "44.369%"
        )

    with col3:

        st.metric(
            "Weighted F1-Score",
            "83.427%"
        )

    st.divider()

    st.markdown(
        "### 📝 Write Your Consumer Complaint"
    )


# =========================================================
# 15. CLASSIFICATION PAGE AND DASHBOARD INPUT
# =========================================================

if page in ["Dashboard", "Classify Complaint"]:

    if page == "Classify Complaint":

        st.markdown(
            '<div class="main-heading">'
            '📝 Classify Consumer Complaint'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="subtitle">'
            'Write your complaint and let ComplaintIQ analyze it.'
            '</div>',
            unsafe_allow_html=True
        )

    complaint = st.text_area(
        "Write Your Complaint",
        key="complaint_input",
        height=170,
        placeholder=(
            "Example: I noticed unauthorized transactions "
            "on my credit card statement..."
        )
    )

    words = len(complaint.split())

    characters = len(complaint)

    col1, col2 = st.columns(2)

    with col1:

        st.caption(
            f"📝 Words: {words}"
        )

    with col2:

        st.caption(
            f"🔢 Characters: {characters}"
        )

    button1, button2 = st.columns([3, 1])

    with button1:

        analyze = st.button(
            "🚀 Analyze Complaint",
            use_container_width=True
        )

    with button2:

        st.button(
            "🗑️ Clear",
            use_container_width=True,
            on_click=clear_complaint
        )

    # Prediction

    if analyze:

        if not complaint.strip():

            st.warning(
                "Please enter a complaint before submitting."
            )

            st.session_state.prediction_result = None

        else:

            try:

                with st.spinner(
                    "Analyzing your complaint..."
                ):

                    prediction, processed_text = (
                        predict_complaint(complaint)
                    )

                    st.session_state.prediction_result = {
                        "category": str(prediction),
                        "processed_text": processed_text
                    }

            except Exception as error:

                st.session_state.prediction_result = None

                st.error(
                    f"Prediction failed: {error}"
                )


# =========================================================
# DISPLAY PREDICTION RESULT
# =========================================================

result = st.session_state.prediction_result

if result:

    st.divider()

    st.markdown("### 📊 Complaint Classification Result")

    # Display result using Streamlit native components
    # instead of embedded HTML.

    st.markdown(
        """
        <style>
        .result-wrapper {
            background: linear-gradient(120deg, #075b4c, #08745a);
            border: 1px solid #159b78;
            border-radius: 15px;
            padding: 30px 20px;
            text-align: center;
            margin-top: 15px;
        }

        .result-emoji {
            font-size: 45px;
        }

        .result-category {
            color: white;
            font-size: 24px;
            font-weight: 800;
            margin-top: 12px;
        }

        .result-message {
            color: #d4f5e9;
            font-size: 14px;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="result-wrapper">
            <div class="result-emoji">📋</div>
            <div class="result-category">
                {html.escape(result["category"])}
            </div>
            <div class="result-message">
                Your complaint has been successfully analyzed
                by the Machine Learning model.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")

    with st.expander("🔍 View Preprocessed Complaint"):

        st.write(result["processed_text"])

    st.caption(
        "Prediction is based on patterns learned from the training dataset."
    )


# =========================================================
# 16. MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="main-heading">'
        '📈 Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Evaluation results of the selected Balanced Linear SVC model.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 📊 Evaluation Metrics"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Accuracy",
            "82.945%"
        )

        st.metric(
            "Macro Precision",
            "48.514%"
        )

        st.metric(
            "Macro Recall",
            "45.821%"
        )

    with col2:

        st.metric(
            "Macro F1-Score",
            "44.369%"
        )

        st.metric(
            "Weighted F1-Score",
            "83.427%"
        )

    st.divider()

    st.markdown(
        "### ⚙️ Selected Model Configuration"
    )

    st.markdown("""
    **Algorithm:** Linear Support Vector Classifier

    **Class Weight:** Balanced

    **Random State:** 42

    **Maximum Iterations:** 3000

    **Feature Extraction:** TF-IDF Unigram

    **Vocabulary Size:** 16,002

    **Number of Model Classes:** 17
    """)

    st.divider()

    st.markdown(
        "### 📌 Performance Interpretation"
    )

    st.write("""
    - Accuracy measures the overall proportion of correct predictions.
    - Macro Precision evaluates precision equally across categories.
    - Macro Recall evaluates recall equally across categories.
    - Macro F1-score balances precision and recall across categories.
    - Weighted F1-score accounts for the number of samples in each category.

    The difference between weighted F1 and macro F1 indicates
    that performance varies across product categories,
    particularly for minority classes.
    """)

    st.info(
        "Balanced Linear SVC was selected based on its "
        "Macro F1-score among the evaluated configurations."
    )


# =========================================================
# 17. ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.markdown(
        '<div class="main-heading">'
        'ℹ️ About ComplaintIQ'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'An end-to-end Natural Language Processing and '
        'Machine Learning project.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 🎯 Project Objective"
    )

    st.write("""
    ComplaintIQ is an NLP-powered Machine Learning application
    designed to classify consumer financial complaints into
    relevant product categories.

    The application analyzes complaint narratives and predicts
    their corresponding financial product category.
    """)

    st.divider()

    st.markdown(
        "### 📚 Dataset Information"
    )

    st.write("""
    **Dataset:** Consumer Financial Protection Bureau (CFPB)

    **Original Records:** 3,093,136

    **Records with Narratives:** 1,111,330

    **Final Cleaned Records:** 970,109

    **Cleaned Dataset Categories:** 18

    **Model Trained Categories:** 17
    """)

    st.divider()

    st.markdown(
        "### 🧠 Machine Learning Workflow"
    )

    workflow = [
        "Data Collection",
        "Exploratory Data Analysis",
        "Text Preprocessing",
        "Stopword Removal",
        "POS-based Lemmatization",
        "TF-IDF Unigram Feature Extraction",
        "Machine Learning Model Training",
        "Class Imbalance Handling",
        "Model Evaluation",
        "Streamlit Application Development"
    ]

    for index, step in enumerate(workflow, start=1):

        st.write(
            f"**{index}.** {step}"
        )

    st.divider()

    st.markdown(
        "### 🛠️ Technologies Used"
    )

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "NLTK",
        "Scikit-learn",
        "TF-IDF",
        "Linear SVC",
        "Streamlit",
        "Joblib"
    ]

    col1, col2, col3 = st.columns(3)

    for index, technology in enumerate(technologies):

        with [col1, col2, col3][index % 3]:

            st.markdown(
                f"- {technology}"
            )

    st.divider()

    st.markdown(
        "### ⚠️ Model Limitations"
    )

    st.write("""
    The model predicts a single category based on patterns
    learned from its training data.

    It may produce incorrect predictions for ambiguous
    complaints, unfamiliar wording, or underrepresented
    product categories.

    Predictions should be reviewed rather than treated
    as official financial complaint decisions.
    """)

    st.divider()

    st.markdown(
        "### 👨‍💻 Developer"
    )

    st.subheader(
        "Sujith Karnati"
    )

    st.write(
        "Machine Learning | Natural Language Processing | Data Science"
    )


# =========================================================
# 18. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#aab6cf;
        padding:15px;
        line-height:1.8;
    ">
        📊 <b>ComplaintIQ</b>
        <br>
        Consumer Complaint Classification System
        <br>
        Built using Python, NLP and Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
