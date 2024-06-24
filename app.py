import os
import re
import spacy
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
from werkzeug.utils import secure_filename

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Load the fine-tuned model and tokenizer
home_directory = os.path.expanduser('~')
save_directory = os.path.join(home_directory, 'fine_tuned_t5_model')
model = T5ForConditionalGeneration.from_pretrained(save_directory)
tokenizer = T5Tokenizer.from_pretrained(save_directory)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def cleanText(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    
    return text.strip()

def removeStopwordsAndLemmatize(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

def prepareAndClean(text):
    text = cleanText(text)
    text = removeStopwordsAndLemmatize(text)
    return text

def generateSummary(article_text):
    cleaned_text = prepareAndClean(article_text)
    input_text = f"summarize: {cleaned_text}"
    
    inputs = tokenizer(input_text, max_length=512, padding='max_length', truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs.input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    st.set_page_config(page_title='PubMed Article Summarizer', page_icon='📝', layout='wide')
    st.title('📝 PubMed Article Summarizer')
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #f0f2f6;
        }
        .sidebar .sidebar-content {
            background: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextArea textarea {
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    option = st.sidebar.radio('Choose an option:', ('Summarize Text', 'Upload and Summarize'))

    if option == 'Summarize Text':
        st.header('Enter Article Text:')
        article_text = st.text_area('Input your article here:', height=200)
        if st.button('Summarize'):
            with st.spinner('Generating summary...'):
                summary = generateSummary(article_text)
            st.subheader('Summary:')
            st.success(summary)

    elif option == 'Upload and Summarize':
        st.header('Upload PubMed File:')
        uploaded_file = st.file_uploader('Choose a file:', type=['txt'])
        if uploaded_file is not None:
            article_text = uploaded_file.read().decode('utf-8')
            with st.spinner('Generating summary...'):
                summary = generateSummary(article_text)
            st.subheader('Summary:')
            st.success(summary)

if __name__ == '__main__':
    main()
