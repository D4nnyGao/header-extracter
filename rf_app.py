import streamlit as st
import pandas as pd
import pymupdf as fitz
import joblib
import re
from bs4 import BeautifulSoup
from collections import Counter
import spacy
import os
import numpy as np
import html
import json

# --- Global Constants & Keyword Set ---
HEADER_KEYWORDS = {
    'summary', 'synopsis', 'primary', 'secondary', 'objective', 'estimand',
    'design', 'schema', 'schedule', 'activities', 'introduction', 'purpose',
    'endpoint', 'participant', 'rationale','recruitment', 'intervention', 'population',
    'selection', 'eligibility', 'criteria', 'inclusion', 'exclusion',
    'lifestyle', 'concomitant', 'dosing', 'administration', 'modification',
    'treatment', 'preparation', 'handling', 'therapy', 'blinding',
    'randomisation', 'discontinuation', 'withdrawal', 'stopping', 'assessment',
    'procedure', 'efficacy', 'screening', 'baseline', 'adverse', 'events',
    'serious', 'pregnancy', 'pharmacokinetics', 'genetics', 'biomarkers',
    'immunogenicity', 'analysis', 'statistical', 'hypothesis', 'interim',
    'sample', 'appendix'
}

# Updated list of features the model expects for prediction
try:
    with open('model_features.json', 'r') as f:
        model_features = json.load(f)['features']
except FileNotFoundError:
    st.error("'model_features.json' not found. Please run the training script to generate it.")
    st.stop()


# --- Helper Functions ---
def get_text_case(text):
    """Determines the case of the text string."""
    if text.isupper(): return "upper"
    if text.islower(): return "lower"
    if text.istitle(): return "title"
    if text and text[0].isupper() and any(c.islower() for c in text[1:]):
        return "sentence"
    return "mixed"

def has_header_pattern(text: str) -> bool:
    """
    Checks if a line of text matches any of the common header patterns.
    Returns True if a pattern is found, False otherwise.
    """
    if not isinstance(text, str):
        return False
    
    line = text.strip()
    if not line:
        return False

    # Combined regex for various header formats (e.g., "1.2", "I.", "A)", "Conclusion:")
    header_patterns_regex = re.compile(
        r"(^\s*\d[\d\.\)]*|"          # Starts with number pattern (1.2, 3))
        r"^\s*[IVXLCDMivxlcdm]+[\.\)]|" # Starts with Roman numeral (I., iv))
        r"^\s*[A-Za-z][\.\)]|"        # Starts with alphabetic pattern (A., (b))
        r".*:$)"                      # Ends with a colon
    )
    return bool(header_patterns_regex.search(line))

def identify_common_elements_full(doc, threshold=0.7):
    """
    Detects recurring headers/footers based on position and content,
    by examining every page of the document.
    """
    if doc.page_count < 2:
        return set()

    text_positions = Counter()
    for page in doc:
        blocks = page.get_text("blocks")
        for x0, y0, x1, y1, text, _, _ in blocks:
            normalized = re.sub(r'[\d\s]+', '', text.strip())
            if len(normalized) > 3:
                pos = round(y0 / 10) * 10
                text_positions[(normalized, pos)] += 1
    
    min_occurrences = max(2, int(doc.page_count * threshold))
    return {item for item, count in text_positions.items() if count >= min_occurrences}


@st.cache_resource
def load_spacy_model():
    """Load the SpaCy model once."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        st.stop()

@st.cache_resource
def load_model():
    """Load the trained Random Forest model once."""
    model_path = 'random_forest_model.joblib'
    if not os.path.exists(model_path):
        st.error(f"Model file not found. Make sure '{model_path}' is in the same directory.")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading the model file: {e}")
        return None


def predict_headers_in_pdf(pdf_file, model):
    """
    Processes an uploaded PDF file, generates all required features for the new model,
    and returns text with header predictions.
    """
    nlp = load_spacy_model()
    all_lines_data = []
    
    keyword_regex = r'\b(?:' + '|'.join(re.escape(word) for word in HEADER_KEYWORDS) + r')\b'

    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    common_elements = identify_common_elements_full(doc)

    font_sizes = []
    for page in doc:
        page_lines = {}
        soup = BeautifulSoup(page.get_text("html"), "html.parser")
        
        for p in soup.find_all('p'):
            style = p.get('style', '')
            top_match = re.search(r'top:([\d\.]+)pt', style)
            if top_match:
                top = float(top_match.group(1))
                left_match = re.search(r'left:([\d\.]+)pt', style)
                left = float(left_match.group(1)) if left_match else 0.0
                
                grouped_top = top
                for existing_top in page_lines.keys():
                    if abs(top - existing_top) <= 2.0:
                        grouped_top = existing_top
                        break
                
                if grouped_top not in page_lines:
                    page_lines[grouped_top] = []
                page_lines[grouped_top].append((left, p))
        
        for top in sorted(page_lines.keys()):
            line_ps = sorted(page_lines[top], key=lambda x: x[0])
            line_text = ' '.join(p_tag.get_text(strip=True) for _, p_tag in line_ps)
            
            if not line_text.strip():
                continue

            normalized_text = re.sub(r'[\d\s]+', '', line_text).strip()
            pos = round(top / 10) * 10
            if (normalized_text, pos) in common_elements:
                continue

            first_p = line_ps[0][1]
            first_span = first_p.find('span')
            font_size = 12.0
            if first_span and first_span.get('style'):
                size_match = re.search(r'font-size:([\d\.]+)pt', first_span['style'])
                if size_match:
                    font_size = float(size_match.group(1))
            
            is_bold = any('<b>' in str(p) or '<strong>' in str(p) for _, p in line_ps)
            
            font_sizes.append(font_size)
            all_lines_data.append({
                'text': line_text,
                'font_size': font_size,
                'is_bold': int(is_bold),
                'char_count': len(line_text)
            })

    if not all_lines_data:
        return []
    
    features_df = pd.DataFrame(all_lines_data)

    most_common_font = Counter(font_sizes).most_common(1)[0][0] if font_sizes else 12.0
    max_len_in_doc = features_df['char_count'].max()

    features_df['font_flag'] = (features_df['font_size'] > most_common_font).astype(int)
    
    if max_len_in_doc > 0:
        features_df['relative_length_ratio'] = features_df['char_count'] / max_len_in_doc
    else:
        features_df['relative_length_ratio'] = 0

    features_df['has_top_header_word'] = features_df['text'].str.contains(
        keyword_regex, case=False, na=False, regex=True
    ).astype(int)

    docs = list(nlp.pipe(features_df['text']))
    features_df['verb_count'] = [sum(1 for token in doc if token.pos_ == 'VERB') for doc in docs]
    features_df['noun_count'] = [sum(1 for token in doc if token.pos_ == 'NOUN') for doc in docs]

    # --- GENERATE THE NEW FEATURE ---
    features_df['has_header_pattern'] = features_df['text'].apply(has_header_pattern).astype(int)

    features_df['text_case'] = features_df['text'].apply(get_text_case)
    df_encoded = pd.get_dummies(features_df, columns=['text_case'], prefix='case') 
    
    # Reindex using the updated feature list to ensure all columns are present
    X_predict = df_encoded.reindex(columns=model_features, fill_value=0)

    predictions = model.predict(X_predict)
    features_df['is_header_prediction'] = predictions
    return features_df.to_dict('records')


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📄 Clinical Protocol Header Detection")
st.markdown("Upload a clinical study protocol PDF to identify section headers using a trained **Random Forest** model.")

model = load_model()

if model:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner('Analyzing PDF and predicting headers... This may take a moment.'):
            results = predict_headers_in_pdf(uploaded_file, model)

        if not results:
            st.warning("Could not extract any processable text from this PDF.")
        else:
            st.success("Processing complete!")
            
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📝 Identified Headers")
                identified_headers = [res['text'] for res in results if res['is_header_prediction'] == 1]
                if identified_headers:
                    # Using html.escape to prevent markdown rendering issues with header text
                    header_md = "<br>".join([f"- {html.escape(h)}" for h in identified_headers])
                    st.markdown(f"<div style='height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>{header_md}</div>", unsafe_allow_html=True)
                else:
                    st.info("No headers were identified based on the model's criteria.")
            
            with col2:
                st.subheader("📑 Full Text with Highlighting")
                full_text_html = ""
                for res in results:
                    text = res['text']
                    escaped_text = html.escape(text)
                    if res['is_header_prediction'] == 1:
                        full_text_html += f"<p style='margin-bottom: 2px;'><mark style='background-color: yellow;'><strong>{escaped_text}</strong></mark></p>"
                    else:
                        full_text_html += f"<p style='margin-bottom: 2px;'>{escaped_text}</p>"
                
                st.markdown(f"<div style='height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>{full_text_html}</div>", unsafe_allow_html=True)