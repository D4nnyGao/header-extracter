import streamlit as st
import pymupdf as fitz
import anthropic
import json
import os
import re  # Import the regular expression module
from dotenv import load_dotenv
from thefuzz import fuzz

# --- ⚙️ CONFIGURATION ---
MODEL_NAME = "claude-sonnet-4-20250514"

# --- 🧠 BACKEND LOGIC (Helper Functions) ---

def _extract_text_with_markers(pdf_bytes: bytes) -> tuple[str | None, list | None, str | None]:
    """
    Internal helper to extract text.
    Returns (llm_text, page_texts, None) on success, or (None, None, error_message) on failure.
    """
    try:
        full_text_for_llm = ""
        pages_text = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                pages_text.append(page_text)
                full_text_for_llm += f"[PAGE {page_num + 1} START]\n{page_text}\n[PAGE {page_num + 1} END]\n\n"
        return full_text_for_llm, pages_text, None
    except Exception as e:
        return None, None, f"Failed to read or process the PDF file. It might be corrupted. Details: {e}"

def _get_headers_from_claude(full_text_for_llm: str, api_key: str) -> tuple[bool, list | str]:
    """
    Internal helper to send document text to Claude.
    Returns (True, data) on success or (False, error_message) on failure.
    """
    if not api_key:
        return False, "Anthropic API key is missing."
        
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        return False, f"Failed to initialize the Anthropic client. Is the API key valid? Details: {e}"
        
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

    try:
        message = client.messages.create(
            model=MODEL_NAME, max_tokens=4096,
            system=system_prompt, messages=[{"role": "user", "content": user_prompt}]
        )
        response_text = message.content[0].text
        print("\n--- 🕵️ Raw LLM Output Received ---")
        print(response_text)
        print("-------------------------------------\n")
        parsed_data = json.loads(response_text)
        headers_data = parsed_data.get("headers", [])
        
        if headers_data and isinstance(headers_data, list) and (len(headers_data) == 0 or isinstance(headers_data[0], dict)):
            return True, headers_data
        else:
            return False, f"LLM returned valid JSON, but in an unexpected format:\n\n{response_text}"

    except json.JSONDecodeError:
        return False, f"Failed to decode JSON. LLM Raw Output:\n\n{response_text}"
    except Exception as e:
        return False, f"An error occurred while calling the Claude API: {e}"

def _chunk_content_by_page(pages_text: list, headers_data: list) -> list:
    """
    Accurately chunks content using fuzzy matching to find the best header candidate on each page.
    """
    header_positions = []
    MATCH_THRESHOLD = 85  # How similar strings need to be (out of 100) to be a match

    for item in headers_data:
        header_text = item.get("header")
        page_num = item.get("page_number")

        if not header_text or not isinstance(page_num, int):
            continue
        
        if 1 <= page_num <= len(pages_text):
            page_content = pages_text[page_num - 1]
            
            best_match_score = 0
            best_match_line = None
            
            for line in page_content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                score = fuzz.partial_ratio(header_text.lower(), line.lower())
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_line = line
            
            if best_match_score >= MATCH_THRESHOLD:
                try:
                    start_pos_on_page = page_content.index(best_match_line)
                    header_positions.append({**item, "start_pos_on_page": start_pos_on_page})
                except ValueError:
                    continue

    header_positions.sort(key=lambda x: (x["page_number"], x["start_pos_on_page"]))

    document_chunks = []
    for i, current_header in enumerate(header_positions):
        start_page_idx = current_header["page_number"] - 1
        start_offset = current_header["start_pos_on_page"]
        
        next_header = header_positions[i + 1] if i + 1 < len(header_positions) else None
        
        if next_header:
            end_page_idx = next_header["page_number"] - 1
            end_offset = next_header["start_pos_on_page"]
        else:
            end_page_idx = len(pages_text) - 1
            end_offset = len(pages_text[-1])

        content_parts = []
        if start_page_idx == end_page_idx:
            content = pages_text[start_page_idx][start_offset:end_offset]
            content_parts.append(content)
        else:
            content_parts.append(pages_text[start_page_idx][start_offset:])
            for page_idx in range(start_page_idx + 1, end_page_idx):
                content_parts.append(pages_text[page_idx])
            content_parts.append(pages_text[end_page_idx][:end_offset])
        
        full_content = "".join(content_parts)
        document_chunks.append({**current_header, "content": full_content.strip()})
        
    return document_chunks

@st.cache_data(show_spinner=False)
def process_pdf(_pdf_bytes: bytes, api_key: str) -> tuple[bool, list | str]:
    """
    Main cached workflow: extract text, call LLM, and chunk content.
    Returns (True, chunk_data) or (False, error_message).
    """
    full_text_for_llm, pages_text, error = _extract_text_with_markers(_pdf_bytes)
    if error:
        return False, error

    success, data = _get_headers_from_claude(full_text_for_llm, api_key)
    if not success:
        return False, data

    headers_data = data
    document_chunks = _chunk_content_by_page(pages_text, headers_data)
    
    return True, document_chunks

# --- 🖥️ STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Protocol Section Viewer")
st.title("📄 Clinical Protocol Section Viewer")
st.markdown("Upload a PDF to identify sections and view their content. The LLM identifies headers and their page numbers for accurate chunking.")

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    st.warning("ANTHROPIC_API_KEY not found in your `.env` file. Please enter it below.", icon="🔑")
    api_key = st.text_input("Enter your Anthropic (Claude) API Key:", type="password")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if not api_key:
        st.error("Please provide your Claude API key to proceed.", icon="🚫")
        st.stop()
    
    with st.spinner("Analyzing document with Claude... This may take a minute for large files."):
        pdf_bytes = uploaded_file.getvalue()
        success, result = process_pdf(pdf_bytes, api_key=api_key)

    if success:
        document_chunks = result
        if not document_chunks:
            st.warning("Analysis complete, but no headers were identified by the model.")
        else:
            st.success(f"Successfully processed and identified {len(document_chunks)} sections!")
            
            for i, chunk in enumerate(document_chunks):
                page_num = chunk.get('page_number', 'N/A')
                header = chunk.get('header', 'Unknown Header')
                expander_title = f"Page {page_num} — {header}"
                
                with st.expander(expander_title):
                    st.text_area("Section Content", chunk['content'], height=300, key=f"exp_{i}_{page_num}")
    else:
        error_message = result
        st.error("Could not process the document.", icon="🚨")
        st.text_area("Error Details", error_message, height=200)