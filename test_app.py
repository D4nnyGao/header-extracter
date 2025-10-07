import streamlit as st
import pymupdf as fitz
import json
import re
from thefuzz import fuzz # <-- Import the fuzzy matching library

# --- Hardcoded Example LLM Output ---
EXAMPLE_LLM_OUTPUT = {
  "headers": [
    { "header": "1. INTRODUCTION AND RATIONALE", "page_number": 7 },
    { "header": "1.1. Current Therapies Ineffective", "page_number": 7 },
    { "header": "1.2. Genomic and Proteomic Technologies", "page_number": 9 },
    { "header": "1.3. Circulating Tumor Cells", "page_number": 9 },
    { "header": "1.4. Immune Cells", "page_number": 11 },
    { "header": "1.5. Hematopoietic Stem and Progenitor Cells", "page_number": 11 },
    { "header": "1.6. Cell Culture Technologies", "page_number": 11 },
    { "header": "1.7. Tumor Resistance and Sensitivity Assays", "page_number": 12 },
    { "header": "1.8. Growth-Based Tumor Resistance Assays", "page_number": 13 },
    { "header": "1.8.1. Mitochondrial Tetrazol Assay (MTT)", "page_number": 13 },
    { "header": "1.9. Tumor Apoptosis-Viability Assays (TAVA)", "page_number": 13 },
    { "header": "1.10. Apoptosis Pathways and Development of Tumor Apoptosis-Viability Assays (TAVA)", "page_number": 14 },
    { "header": "1.11 Anti-tumor Lymphocyte Cytotoxicity Assay (Cytotoxicity Assay)", "page_number": 15 },
    { "header": "1.12 COVID-19 Research", "page_number": 15 },
    { "header": "2. STUDY DESIGN", "page_number": 16 },
    { "header": "2.1. Objectives", "page_number": 16 },
    { "header": "2.1.1 Primary Objective", "page_number": 16 },
    { "header": "2.1.2 Secondary Objectives", "page_number": 16 },
    { "header": "2.2 Endpoints", "page_number": 17 },
    { "header": "2.2.1 Primary Endpoint", "page_number": 17 },
    { "header": "2.2.2 Secondary Endpoint", "page_number": 17 },
    { "header": "2.3. Selection of Study Population", "page_number": 17 },
    { "header": "2.3.1. Study Population & Sample Size", "page_number": 18 },
    { "header": "2.3.2. Study Duration", "page_number": 20 },
    { "header": "2.3.3. Inclusion Criteria", "page_number": 20 },
    { "header": "2.3.4. Exclusion Criteria", "page_number": 23 },
    { "header": "3. STUDY METHODS", "page_number": 23 },
    { "header": "3.1 Assignment (Enrollment) to Study", "page_number": 23 },
    { "header": "3.2. Clinical Procedures", "page_number": 24 },
    { "header": "3.2.1. Procedures by Study Visit", "page_number": 24 },
    { "header": "3.2.2. Mobilizing Immune Cells with Exercise", "page_number": 27 },
    { "header": "3.2.3. Safety Parameters", "page_number": 28 },
    { "header": "3.2.4. Tumor Status", "page_number": 29 },
    { "header": "3.2.5. ECOG Performance Status", "page_number": 29 },
    { "header": "3.2.6. Concomitant Anticancer Therapy Documentation", "page_number": 30 },
    { "header": "3.2.7. Blood Draw for Circulating Tumor Cells and Immune Cells", "page_number": 30 },
    { "header": "3.2.8. Biopsies", "page_number": 30 },
    { "header": "3.2.9. Leukapheresis Collection for Circulating Tumor Cells, Immune Cells and Progenitor Cells", "page_number": 30 },
    { "header": "3.2.10. Microaggregate Filter for CTC Collection During Leukapheresis", "page_number": 31 },
    { "header": "3.2.11. Long Term Follow-Up Period", "page_number": 33 },
    { "header": "3.2.12. Study Completion, Follow-up, and Close Out", "page_number": 33 },
    { "header": "3.2.13. Subject Withdrawal", "page_number": 34 },
    { "header": "3.2.14 Possible Toxicities", "page_number": 34 },
    { "header": "3.3. Laboratory Procedures", "page_number": 36 },
    { "header": "3.3.1. Processing and Isolation of Tumor, Circulating Tumor Cells, Immune Cells and Progenitor Cells", "page_number": 36 },
    { "header": "3.3.2. Harvesting of Circulating Tumor Cells, Immune Cells and Progenitor Cells", "page_number": 37 },
    { "header": "3.3.3. Cryopreservation of Circulating Tumor Cells, Immune Cells and Progenitor Cells", "page_number": 37 },
    { "header": "3.3.4. Storage of Plasma for Biomarker Discovery", "page_number": 37 },
    { "header": "4. ADVERSE EVENT REPORTING", "page_number": 37 },
    { "header": "5. STATISTICAL METHODS", "page_number": 38 },
    { "header": "5.1. Sample Size Determination", "page_number": 38 },
    { "header": "5.2. Statistical Consideration", "page_number": 38 },
    { "header": "5.3. Statistical Analysis", "page_number": 38 },
    { "header": "6. VALIDATION OF TUMOR APOPTOSIS-VIABILITY ASSAY (TAVA) & LYMPHOCYTE TUMOR CYTOTOXICITY ASSAYS (LTCA)", "page_number": 38 },
    { "header": "7. ETHICAL ASPECTS", "page_number": 39 },
    { "header": "7.1. Compliance", "page_number": 39 },
    { "header": "7.1.1. Good Clinical Practice", "page_number": 39 },
    { "header": "7.1.2. Good Laboratory Practices (GLP) and Good Tissue Practices (GTP)", "page_number": 39 },
    { "header": "7.2. Institutional Review Board/Independent Ethics Committee", "page_number": 39 },
    { "header": "7.3. Informed Consent", "page_number": 39 },
    { "header": "7.4. Financial Disclosure", "page_number": 40 },
    { "header": "7.5. Conflicts of Interest", "page_number": 41 },
    { "header": "7.6. Protection and Monitoring of Conflicts of Interest", "page_number": 41 },
    { "header": "8. BIOREPOSITORY FUNCTION AND GUIDELINES", "page_number": 41 },
    { "header": "8.1. Biospecimen Processing, Storage and Retrieval", "page_number": 41 },
    { "header": "8.2. Quality Assurance/Quality Control", "page_number": 41 },
    { "header": "8.3. Biosafety", "page_number": 42 },
    { "header": "8.4. Biorepository Informatics", "page_number": 42 },
    { "header": "8.5. Privacy Protection and Security", "page_number": 42 },
    { "header": "8.6. Custodianship and Intellectual Property", "page_number": 42 },
    { "header": "9. ADMINISTRATIVE REQUIREMENTS", "page_number": 43 },
    { "header": "9.1. Protocol Amendments", "page_number": 43 },
    { "header": "9.2. Monitoring Procedures", "page_number": 43 },
    { "header": "9.3. Recording of Data and Retention of Documents", "page_number": 44 },
    { "header": "9.4. Auditing Procedures", "page_number": 45 },
    { "header": "9.5. Handling of Anonymous Tumor Specimens Beyond Study Closure", "page_number": 45 },
    { "header": "9.6. Publication of Results", "page_number": 45 },
    { "header": "9.7. Disclosure and Confidentiality", "page_number": 46 },
    { "header": "9.8. Discontinuation of Study", "page_number": 46 },
    { "header": "9.9. Data Management", "page_number": 46 },
    { "header": "9.9.1. Data Collection", "page_number": 46 },
    { "header": "9.9.2. Data Management and Quality Control", "page_number": 46 },
    { "header": "10. APPENDICES", "page_number": 47 },
    { "header": "10.1. Appendix 1: Time and Events Schedule", "page_number": 47 },
    { "header": "10.2. Appendix 2: ECOG Performance Status", "page_number": 48 },
    { "header": "10.3. Appendix 3: First-Generation Guidelines for NCI-Supported Biorepositories", "page_number": 49 },
    { "header": "10.4. Appendix 4: Guidance on Informed Consent for In Vitro Diagnostic Device Studies Using Leftover Human Specimens that are Not Individually Identifiable", "page_number": 49 },
    { "header": "10.5. Appendix 5: FDA Approval of In-Vitro Diagnostic Circulating Tumor Cell Enumeration via Immunicon Cell Tracks Analyzer II", "page_number": 49 },
    { "header": "10.6 Appendix 6: FDA Guidance on Implementation on Acceptable Full-length and Abbreviated Donor History Questionnaires and Accompanying Materials for Use in Screening Donors of Blood and Blood Components (May 20, 2020)", "page_number": 49 },
    { "header": "10.7 Appendix 7: Cancer and Healthy Volunteer Cohort Flow-gram for Collection of Biospecimen(s) versus Leukapheresis Collection", "page_number": 50 },
    { "header": "11. REFERENCES", "page_number": 50 }
  ]
}

# --- 🧠 BACKEND LOGIC (Helper Functions) ---

def _extract_text_by_page(pdf_bytes: bytes) -> tuple[list | None, str | None]:
    """Extracts text from each page of a PDF."""
    try:
        pages_text = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                pages_text.append(page.get_text("text"))
        return pages_text, None
    except Exception as e:
        return None, f"Failed to read or process the PDF file. Details: {e}"

# ------------------- MODIFIED FUNCTION STARTS HERE -------------------
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
            
            # --- NEW FUZZY MATCHING LOGIC ---
            best_match_score = 0
            best_match_line = None
            
            # Iterate through each line of text on the page
            for line in page_content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Calculate the similarity ratio between the expected header and the current line
                # fuzz.partial_ratio is good for finding a phrase within a longer line
                score = fuzz.partial_ratio(header_text.lower(), line.lower())
                
                # If this line is a better match than any we've seen before, save it
                if score > best_match_score:
                    best_match_score = score
                    best_match_line = line
            
            # After checking all lines, if our best match is good enough, use it.
            if best_match_score >= MATCH_THRESHOLD:
                try:
                    # Find the position of our best matching line in the full page text
                    start_pos_on_page = page_content.index(best_match_line)
                    header_positions.append({**item, "start_pos_on_page": start_pos_on_page})
                except ValueError:
                    # This should rarely happen, but it's a safeguard
                    continue
            # --- END NEW LOGIC ---

    # Sort headers by their actual position in the document to ensure correct order
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
# -------------------- MODIFIED FUNCTION ENDS HERE --------------------

# --- 🖥️ STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="PDF Sectioning Test App")
st.title("🧪 PDF Sectioning Test App (Fuzzy Matching)")
st.markdown("Upload a PDF to test the sectioning logic. This version uses **fuzzy string matching** to locate headers.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text and sectioning document..."):
        pdf_bytes = uploaded_file.getvalue()
        
        pages_text, error = _extract_text_by_page(pdf_bytes)
        
        if error:
            st.error(error)
            st.stop()
            
        headers_data = EXAMPLE_LLM_OUTPUT.get("headers", [])
        document_chunks = _chunk_content_by_page(pages_text, headers_data)

    # --- Display Results ---
    total_headers_to_find = len(headers_data)
    found_headers_count = len(document_chunks)
    
    st.info(f"**Summary:** Found **{found_headers_count}** out of **{total_headers_to_find}** predefined headers.")
    
    if found_headers_count < total_headers_to_find:
        found_headers_set = {chunk['header'] for chunk in document_chunks}
        missed_headers = [item for item in headers_data if item['header'] not in found_headers_set]
        with st.expander(f"⚠️ Click here to see the {len(missed_headers)} headers that were not found"):
            for item in missed_headers:
                st.write(f"- `{item['header']}` (Expected on page {item['page_number']})")

    if not document_chunks:
        st.warning("No matching sections were found. Try adjusting the MATCH_THRESHOLD in the code if needed.")
    else:
        st.success("Displaying found sections below:")
        
        for i, chunk in enumerate(document_chunks):
            page_num = chunk.get('page_number', 'N/A')
            header = chunk.get('header', 'Unknown Header')
            expander_title = f"Page {page_num} — {header}"
            
            with st.expander(expander_title):
                st.text_area("Section Content", chunk['content'], height=300, key=f"exp_{i}_{page_num}")