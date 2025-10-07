import os
import json
import pandas as pd
import anthropic
import fitz  # PyMuPDF
from dotenv import load_dotenv
import re
import time

# --- File Paths ---
GROUND_TRUTH_CSV = "final_labeled_dataset.csv"
PDF_DIRECTORY = "protocols/downloaded_pdfs"

# --- Evaluation Settings ---
# How many random documents from your dataset to test against.
NUM_DOCS_TO_TEST = 20

# --- LLM Settings ---
MODEL_NAME = "claude-sonnet-4-20250514" # Or your preferred model
# ==============================================================================

def _extract_text_with_markers(pdf_bytes: bytes) -> tuple[str | None, str | None]:
    """Extracts text and returns (llm_text, error_message)."""
    try:
        full_text_for_llm = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                full_text_for_llm += f"[PAGE {page_num + 1} START]\n{page_text}\n[PAGE {page_num + 1} END]\n\n"
        return full_text_for_llm, None
    except Exception as e:
        return None, f"Failed to read PDF. Details: {e}"

def _get_headers_from_claude(full_text_for_llm: str, api_key: str) -> tuple[list | None, str | None]:
    """Sends text to Claude and returns (headers_list, error_message)."""
    try:
        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = """
            You are an expert document analyst specializing in clinical trial protocols. Your primary task is to identify and extract all section headers and the page number on which each header appears.

            You MUST return your response as a valid JSON object with a single key: "headers".
            The value for "headers" MUST be a JSON list of objects. Each object must contain two keys:
            1. "header": The string of the section header.
            2. "page_number": The integer page number where the header was found.

            Here is an example of the required output format:
            {
            "headers": [
                { "header": "1. INTRODUCTION", "page_number": 5 },
                { "header": "1.1 Study Rationale", "page_number": 5 },
                { "header": "2. STUDY OBJECTIVES", "page_number": 6 }
            ]
            }

            Your entire response must be ONLY the valid JSON object. Do not include any other text.
        """
        user_prompt = f"Here is the document text:\n---\n{full_text_for_llm}\n---"
        response_text = ""

        message = client.messages.create(
            model=MODEL_NAME, max_tokens=4096,
            system=system_prompt, messages=[{"role": "user", "content": user_prompt}]
        )
        response_text = message.content[0].text
        headers_data = json.loads(response_text).get("headers", [])
        
        # We only need the header text strings for this evaluation
        header_strings = [item.get("header", "") for item in headers_data]
        return [h for h in header_strings if h], None

    except Exception as e:
        return None, f"Claude API Error: {e}"

def normalize_text(text):
    """Simple normalization for accurate comparison."""
    return re.sub(r'\s+', ' ', text).strip().lower()

# --- MAIN EVALUATION SCRIPT ---
def main():
    print("Starting LLM Header Extraction Evaluation...")
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in `.env` file. Exiting.")
        return

    # --- 1. Load Ground Truth Data ---
    try:
        print(f"Loading ground truth data from '{GROUND_TRUTH_CSV}'...")
        df_truth = pd.read_csv(GROUND_TRUTH_CSV)
    except FileNotFoundError:
        print(f"ERROR: Ground truth file '{GROUND_TRUTH_CSV}' not found. Exiting.")
        return

    # --- 2. Select Documents for Testing ---
    all_doc_ids = df_truth['document_id'].unique()
    if len(all_doc_ids) < NUM_DOCS_TO_TEST:
        print(f"Warning: Found only {len(all_doc_ids)} unique documents. Testing on all of them.")
        docs_to_test = all_doc_ids
    else:
        docs_to_test = pd.Series(all_doc_ids).sample(n=NUM_DOCS_TO_TEST, random_state=42).tolist()
    
    print(f"Selected {len(docs_to_test)} documents to evaluate.\n")

    total_tp, total_fp, total_fn = 0, 0, 0

    # --- 3. Process Each Document and Compare ---
    for i, doc_id in enumerate(docs_to_test):
        print(f"--- Processing Document {i+1}/{len(docs_to_test)}: {doc_id} ---")
        pdf_path = os.path.join(PDF_DIRECTORY, doc_id)
        
        if not os.path.exists(pdf_path):
            print(f"  - PDF not found at '{pdf_path}'. Skipping.")
            continue

        # Get LLM Predictions
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        llm_text, error = _extract_text_with_markers(pdf_bytes)
        if error:
            print(f"  - Error extracting text: {error}. Skipping.")
            continue
            
        predicted_headers, error = _get_headers_from_claude(llm_text, api_key)
        if error:
            print(f"  - Error calling Claude API: {error}. Skipping.")
            # Add a delay to avoid hammering the API if there's a persistent issue
            time.sleep(5)
            continue
        
        # Normalize LLM predictions
        predicted_set = {normalize_text(h) for h in predicted_headers}
        print(f"  - LLM Predicted {len(predicted_set)} unique headers.")

        # Get Ground Truth Headers
        doc_truth_df = df_truth[df_truth['document_id'] == doc_id]
        actual_headers = doc_truth_df[doc_truth_df['is_header'] == 1]['text'].tolist()
        actual_set = {normalize_text(h) for h in actual_headers}
        print(f"  - Ground Truth has {len(actual_set)} unique headers.")
        
        # Calculate Metrics for this document
        tp = len(predicted_set.intersection(actual_set))
        fp = len(predicted_set - actual_set)
        fn = len(actual_set - predicted_set)
        
        print(f"  - Results: TP={tp}, FP={fp}, FN={fn}")

        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Optional delay to respect API rate limits
        time.sleep(1) 

    # --- 4. Calculate Final Scores ---
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    
    print(f"Total Documents Tested: {len(docs_to_test)}")
    print(f"Total True Positives (TP): {total_tp}")
    print(f"Total False Positives (FP): {total_fp}")
    print(f"Total False Negatives (FN): {total_fn}")
    
    # Calculate Precision, Recall, and F1 Score
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- METRICS ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1_score:.4f}")
   
if __name__ == "__main__":
    main()