# --- IMPORTS ---
import csv
import requests
import os
import json
import re
import pymupdf as fitz
import pandas as pd
from bs4 import BeautifulSoup
import spacy
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import numpy as np



INPUT_CSV_PATH = 'protocols/csv/ctg-studies.csv' # Path to the CSV with NCT numbers
DOWNLOAD_DIR = "protocols/downloaded_pdfs"      # Where to save the downloaded PDFs

PDF_DIRECTORY = DOWNLOAD_DIR                  # Use the same directory for processing
OUTPUT_CSV = "final_labeled_dataset.csv"      # The final output file
SPACY_MODEL = "en_core_web_sm"                # spaCy model for NLP
# Use one less than the total number of CPU cores to keep the computer responsive
NUM_PROCESSES = max(1, cpu_count() - 1)

# Keyword set updated to match the Streamlit application
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
# ==============================================================================

# --- DOWNLOADER ---
def download_protocols(csv_path, output_dir):
    """
    Downloads study protocol documents from a CSV file.
    """
    print("--- Stage 1: Downloading Protocols ---")
    if not os.path.exists(csv_path):
        print(f"ERROR: Input CSV not found at '{csv_path}'. Cannot start download.")
        return False
        
    os.makedirs(output_dir, exist_ok=True)
    failed_downloads = []
    
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            nct_index = header.index('NCT Number')
            docs_index = header.index('Study Documents')

            for row in reader:
                nct_number = row[nct_index]
                study_docs_str = row[docs_index]
                if not study_docs_str: continue

                documents = study_docs_str.split('|')
                latest_protocol, url_match = None, None
                for doc in reversed(documents):
                    current_url_match = re.search(r'https://\S+', doc)
                    if current_url_match and re.search(r'Protocol', doc, re.IGNORECASE):
                        latest_protocol, url_match = doc, current_url_match
                        break

                if latest_protocol and url_match:
                    try:
                        doc_url = url_match.group(0).strip()
                        file_name = f"{nct_number}_protocol.pdf"
                        file_path = os.path.join(output_dir, file_name)
                        
                        if os.path.exists(file_path):
                            print(f"Skipping {nct_number}: Already downloaded.")
                            continue

                        print(f"Downloading for {nct_number}...")
                        response = requests.get(doc_url, stream=True, timeout=30)
                        response.raise_for_status()

                        with open(file_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                    except requests.exceptions.RequestException as e:
                        failed_downloads.append(f"{nct_number}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")

    if failed_downloads:
        print("\n--- Failed Downloads ---")
        for fail in failed_downloads:
            print(fail)
    print("Download stage complete.")
    return True

# --- FEATURE EXTRACTOR HELPERS (MATCHING STREAMLIT APP) ---
def get_text_case(text):
    if text.isupper(): return "upper"
    if text.islower(): return "lower"
    if text.istitle(): return "title"
    if text and text[0].isupper() and any(c.islower() for c in text[1:]): return "sentence"
    return "mixed"

def has_header_pattern(text: str) -> bool:
    if not isinstance(text, str) or not text.strip(): return False
    header_patterns_regex = re.compile(
        r"(^\s*\d[\d\.\)]*|^\s*[IVXLCDMivxlcdm]+[\.\)]|^\s*[A-Za-z][\.\)]|.*:$)"
    )
    return bool(header_patterns_regex.search(text.strip()))

def quick_header_check(text, toc_titles):
    if not toc_titles: return 0
    text_clean = ''.join(text.lower().split())
    if not text_clean: return 0
    for toc_title in toc_titles:
        toc_clean = ''.join(toc_title.lower().split())
        if not toc_clean: continue
        if toc_clean in text_clean or text_clean in toc_clean: return 1
    return 0

def identify_common_elements_full(doc, threshold=0.7):
    if doc.page_count < 2: return set()
    text_positions = Counter()
    for page in doc:
        for _, y0, _, _, text, _, _ in page.get_text("blocks"):
            normalized = re.sub(r'[\d\s]+', '', text.strip())
            if len(normalized) > 3:
                pos = round(y0 / 10) * 10
                text_positions[(normalized, pos)] += 1
    min_occurrences = max(2, int(doc.page_count * threshold))
    return {item for item, count in text_positions.items() if count >= min_occurrences}

# --- MAIN WORKER FUNCTION ---
def process_pdf(pdf_path, spacy_model):
    """
    Processes a single PDF to extract all features, matching the Streamlit app's logic.
    """
    doc_id = os.path.basename(pdf_path)
    print(f"Processing: {doc_id}")
    try:
        doc = fitz.open(pdf_path)
        
        # Get Table of Contents for ground truth labels
        toc_by_page = {}
        for _, title, page_num in doc.get_toc():
            page_idx = page_num - 1
            if page_idx not in toc_by_page: toc_by_page[page_idx] = []
            toc_by_page[page_idx].append(title.strip())
        
        common_elements = identify_common_elements_full(doc)
        all_lines_data, font_sizes = [], []

        # 1. Parse PDF text and basic styles (same as Streamlit)
        for page_num, page in enumerate(doc):
            soup = BeautifulSoup(page.get_text("html"), "html.parser")
            page_lines = {}
            for p in soup.find_all('p'):
                style = p.get('style', '')
                top_match = re.search(r'top:([\d\.]+)pt', style)
                if top_match:
                    top, left = float(top_match.group(1)), 0.0
                    left_match = re.search(r'left:([\d\.]+)pt', style)
                    if left_match: left = float(left_match.group(1))
                    grouped_top = next((et for et in page_lines if abs(top - et) <= 2.0), top)
                    if grouped_top not in page_lines: page_lines[grouped_top] = []
                    page_lines[grouped_top].append((left, p))
            
            for top in sorted(page_lines.keys()):
                line_ps = sorted(page_lines[top], key=lambda x: x[0])
                line_text = ' '.join(p_tag.get_text(strip=True) for _, p_tag in line_ps)
                if not line_text.strip(): continue

                normalized_text = re.sub(r'[\d\s]+', '', line_text).strip()
                pos = round(top / 10) * 10
                if (normalized_text, pos) in common_elements: continue

                font_size = 12.0
                first_span = line_ps[0][1].find('span')
                if first_span and 'style' in first_span.attrs:
                    size_match = re.search(r'font-size:([\d\.]+)pt', first_span['style'])
                    if size_match: font_size = float(size_match.group(1))
                font_sizes.append(font_size)
                
                all_lines_data.append({
                    'document_id': doc_id, 'page': page_num + 1, 'text': line_text,
                    'font_size': font_size, 'char_count': len(line_text),
                    'is_bold': any('<b>' in str(p) or '<strong>' in str(p) for _, p in line_ps),
                    'is_italic': any('<i>' in str(p) or '<em>' in str(p) for _, p in line_ps),
                    'is_underlined': any('<u>' in str(p) for _, p in line_ps),
                    'is_header': quick_header_check(line_text, toc_by_page.get(page_num, [])),
                })
        doc.close()

        if not all_lines_data: return None
        
        # Create DataFrame and calculate document-level features
        df = pd.DataFrame(all_lines_data)
        most_common_font = Counter(font_sizes).most_common(1)[0][0] if font_sizes else 12.0
        max_len_in_doc = df['char_count'].max()

        df['font_flag'] = (df['font_size'] > most_common_font).astype(int)
        df['relative_length_ratio'] = df['char_count'] / max_len_in_doc if max_len_in_doc > 0 else 0
        
        # Calculate all remaining line-level features
        keyword_regex = r'\b(?:' + '|'.join(re.escape(word) for word in HEADER_KEYWORDS) + r')\b'
        df['has_top_header_word'] = df['text'].str.contains(keyword_regex, case=False, regex=True).astype(int)
        df['has_header_pattern'] = df['text'].apply(has_header_pattern).astype(int)
        df['text_case'] = df['text'].apply(get_text_case)
        df['word_count'] = df['text'].str.split().str.len()
        df['number_count'] = df['text'].str.count(r'\d')
        df['starts_with_number'] = df['text'].str.match(r'^\s*\d').astype(int)

        # Perform NLP processing
        docs = list(spacy_model.pipe(df['text']))
        df['verb_count'] = [sum(1 for token in doc if token.pos_ == 'VERB') for doc in docs]
        df['noun_count'] = [sum(1 for token in doc if token.pos_ == 'NOUN') for doc in docs]
        
        # Finalize columns and return
        final_cols = ['document_id','page','text','is_header','is_bold','is_italic','is_underlined',
                      'font_flag','starts_with_number','number_count','verb_count','noun_count',
                      'word_count','text_case','has_top_header_word','char_count',
                      'relative_length_ratio','has_header_pattern']
                      
        return df[final_cols]

    except Exception as e:
        print(f"FAILED to process {doc_id}. Reason: {e}")
        return None


if __name__ == "__main__":
    # --- Downloader ---
    if not download_protocols(INPUT_CSV_PATH, DOWNLOAD_DIR):
        # Exit if download fails or source CSV is not found
        exit()

    # --- Run Feature Extraction ---
    print("\n--- Stage 2: Feature Extraction ---")
    start_time = time.time()
    
    # Pre-load the spaCy model once
    print("Loading spaCy model...")
    nlp = spacy.load(SPACY_MODEL)

    pdf_files = [os.path.join(PDF_DIRECTORY, f) for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{PDF_DIRECTORY}'. Aborting.")
        exit()
        
    print(f"Found {len(pdf_files)} PDFs to process with {NUM_PROCESSES} cores...")
    
    # Create a partial function to pass the loaded spaCy model to each worker
    worker_func = partial(process_pdf, spacy_model=nlp)

    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(worker_func, pdf_files)

    # Combine results and save to CSV
    final_df = pd.concat([res for res in results if res is not None], ignore_index=True)
    
    if not final_df.empty:
        final_df.to_csv(OUTPUT_CSV, index=False)
        print("\n" + "="*50)
        print("PIPELINE COMPLETE")
        print(f"Successfully processed {final_df['document_id'].nunique()} documents.")
        print(f"Final dataset with {len(final_df):,} rows saved to: '{OUTPUT_CSV}'")
    else:
        print("\Pipeline finished, but no data was processed successfully.")
        
    end_time = time.time()
    print(f"Total feature extraction time: {end_time - start_time:.2f} seconds.")
    print("="*50)