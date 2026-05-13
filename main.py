import html
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import ollama
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from pypdf import PdfReader


MODEL_NAME = "phi3"
MAX_TEXT_CHARS = 6000
UPLOAD_DIR = "uploads"

app = FastAPI(
    title="AI PDF Analyzer",
    description="Local AI academic paper analyzer with Ollama, Crossref DOI checking, HTML reporting, and reference validation",
    version="1.2.0"
)


@app.get("/")
def home():
    return {
        "message": "Local AI Academic Assistant Running",
        "version": "1.2.0",
        "endpoints": [
            "/health",
            "/system-info",
            "/analyze-text",
            "/check-reference",
            "/check-pdf-dois",
            "/upload-pdf",
            "/upload-pdf-html"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "model": MODEL_NAME
    }


@app.get("/system-info")
def system_info():
    return {
        "application": "AI PDF Analyzer",
        "version": "1.2.0",
        "llm_model": MODEL_NAME,
        "features": [
            "PDF text extraction",
            "AI academic analysis with Ollama",
            "DOI extraction",
            "Crossref DOI validation",
            "Suspicious reference detection",
            "Possible hallucination flagging",
            "APA-style reference generation",
            "IEEE-style reference generation",
            "Duplicate DOI detection",
            "HTML report generation"
        ]
    }


def save_upload_file(file: UploadFile) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    safe_filename = os.path.basename(file.filename or "uploaded_file.pdf")
    unique_filename = f"{uuid.uuid4()}_{safe_filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    return file_path


async def write_upload_to_disk(file: UploadFile, file_path: str) -> None:
    with open(file_path, "wb") as f:
        f.write(await file.read())


def delete_temp_file(file_path: str) -> None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass


def extract_pdf_text(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        raise ValueError(f"PDF read error: {str(e)}")

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text


def extract_basic_metadata(text: str) -> Dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    estimated_title = lines[0] if lines else "Unknown"

    return {
        "estimated_title": estimated_title[:200]
    }


def extract_dois(text: str) -> List[str]:
    doi_pattern = r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b"
    dois = re.findall(doi_pattern, text, re.IGNORECASE)

    cleaned_dois = []
    for doi in dois:
        doi = doi.strip()
        doi = doi.rstrip(".,;:)")
        doi = doi.replace(" ", "")

        if doi and doi.lower() not in [d.lower() for d in cleaned_dois]:
            cleaned_dois.append(doi)

    return cleaned_dois


def find_duplicate_dois(dois: List[str]) -> List[str]:
    seen = set()
    duplicates = []

    for doi in dois:
        doi_lower = doi.lower()
        if doi_lower in seen:
            duplicates.append(doi)
        else:
            seen.add(doi_lower)

    return duplicates


def get_crossref_year(item: Dict[str, Any]) -> Optional[int]:
    if "published-print" in item:
        return item["published-print"]["date-parts"][0][0]
    if "published-online" in item:
        return item["published-online"]["date-parts"][0][0]
    if "published" in item:
        return item["published"]["date-parts"][0][0]
    if "created" in item:
        return item["created"]["date-parts"][0][0]
    return None


def get_crossref_authors(item: Dict[str, Any]) -> List[str]:
    authors = []

    for author in item.get("author", []):
        given = author.get("given", "")
        family = author.get("family", "")
        full_name = f"{given} {family}".strip()

        if full_name:
            authors.append(full_name)

    return authors


def check_crossref_by_doi(doi: str) -> Dict[str, Any]:
    url = f"https://api.crossref.org/works/{doi}"

    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "AI-PDF-Analyzer/1.2.0 (mailto:example@example.com)"}
        )

        if response.status_code == 404:
            return {
                "doi": doi,
                "status": "NOT_FOUND",
                "classification": "SUSPICIOUS",
                "message": "DOI was not found in Crossref."
            }

        response.raise_for_status()
        item = response.json().get("message", {})

        title = item.get("title", [""])[0] if item.get("title") else ""
        publisher = item.get("publisher", "")
        year = get_crossref_year(item)
        authors = get_crossref_authors(item)

        return {
            "doi": doi,
            "status": "VALID_DOI",
            "classification": "VALID",
            "crossref_title": title,
            "authors": authors,
            "year": year,
            "publisher": publisher,
            "crossref_url": f"https://doi.org/{doi}"
        }

    except Exception as e:
        return {
            "doi": doi,
            "status": "ERROR",
            "classification": "SUSPICIOUS",
            "message": str(e)
        }


def check_crossref_by_title(title: str) -> Dict[str, Any]:
    url = "https://api.crossref.org/works"
    params = {
        "query.title": title,
        "rows": 1
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15,
            headers={"User-Agent": "AI-PDF-Analyzer/1.2.0 (mailto:example@example.com)"}
        )
        response.raise_for_status()
        data = response.json()

        items = data.get("message", {}).get("items", [])

        if not items:
            return {
                "input_title": title,
                "status": "NOT_FOUND",
                "classification": "SUSPICIOUS",
                "message": "No matching reference found in Crossref."
            }

        item = items[0]

        crossref_title = item.get("title", [""])[0] if item.get("title") else ""
        doi = item.get("DOI", "")
        publisher = item.get("publisher", "")
        year = get_crossref_year(item)
        authors = get_crossref_authors(item)

        status = "POSSIBLE_MATCH"
        classification = "SUSPICIOUS"

        if title.lower().strip() == crossref_title.lower().strip():
            status = "EXACT_TITLE_MATCH"
            classification = "VALID"

        return {
            "input_title": title,
            "status": status,
            "classification": classification,
            "crossref_title": crossref_title,
            "doi": doi,
            "authors": authors,
            "year": year,
            "publisher": publisher,
            "crossref_url": f"https://doi.org/{doi}" if doi else None
        }

    except Exception as e:
        return {
            "input_title": title,
            "status": "ERROR",
            "classification": "SUSPICIOUS",
            "message": str(e)
        }


def classify_reference_status(result: Dict[str, Any]) -> str:
    if result.get("status") == "VALID_DOI":
        return "VALID"

    doi = result.get("doi", "")

    if not doi or len(doi) < 10:
        return "POSSIBLE_HALLUCINATION"

    if result.get("status") in ["NOT_FOUND", "ERROR"]:
        return "SUSPICIOUS"

    return "SUSPICIOUS"


def extract_references_section(text: str) -> str:
    patterns = [
        r"\bReferences\b\s*(.*)",
        r"\bREFERENCES\b\s*(.*)",
        r"\bBibliography\b\s*(.*)",
        r"\bBIBLIOGRAPHY\b\s*(.*)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    return ""


def extract_reference_candidates(text: str, max_references: int = 10) -> List[str]:
    references_section = extract_references_section(text)

    if not references_section:
        return []

    lines = references_section.splitlines()
    candidates = []

    for line in lines:
        clean_line = line.strip()

        if len(clean_line) < 40:
            continue

        clean_line = re.sub(r"^\[\d+\]\s*", "", clean_line)
        clean_line = re.sub(r"^\d+\.\s*", "", clean_line)

        candidates.append(clean_line)

        if len(candidates) >= max_references:
            break

    return candidates


def format_apa_reference(ref: Dict[str, Any]) -> str:
    authors = ref.get("authors", [])
    title = ref.get("crossref_title", "Unknown title")
    year = ref.get("year", "n.d.")
    doi = ref.get("doi", "")

    if authors:
        author_text = ", ".join(authors[:6])
        if len(authors) > 6:
            author_text += ", et al."
    else:
        author_text = "Unknown author"

    return f"{author_text}. ({year}). {title}. https://doi.org/{doi}"


def format_ieee_reference(ref: Dict[str, Any], index: int) -> str:
    authors = ref.get("authors", [])
    title = ref.get("crossref_title", "Unknown title")
    year = ref.get("year", "n.d.")
    doi = ref.get("doi", "")

    if authors:
        author_text = ", ".join(authors[:3])
        if len(authors) > 3:
            author_text += ", et al."
    else:
        author_text = "Unknown author"

    return f"[{index}] {author_text}, \"{title},\" {year}. doi: {doi}"


def create_structured_prompt(text: str) -> str:
    return f"""
Analyze this academic paper and produce a clear structured academic report.

Use this exact structure:

# Academic Paper Analysis Report

## 1. Paper Identification
- Title:
- Research field:
- Main topic:

## 2. Summary
Write a clear academic summary.

## 3. Methodology
Explain research design, tools, technologies, dataset, implementation, and evaluation procedure.

## 4. Key Findings
List the most important findings.

## 5. Strengths
Identify the main strengths of the paper.

## 6. Limitations
Identify methodological, technical, empirical, and generalization limitations.

## 7. Possible Unsupported Claims
Identify claims that require additional evidence or external verification.

## 8. Possible Hallucinations
Identify any claims, citations, references, or statements that may be hallucinated, unsupported, unverifiable, or too strong.

## 9. Plagiarism Risk Analysis
Assess whether the text contains generic, repetitive, copied-sounding, or insufficiently paraphrased academic language.
Use only LOW / MODERATE / HIGH RISK.
Important: This is not a formal plagiarism detection result.

## 10. AI-Generated Text Risk
Assess whether the paper contains overly generic phrasing, repetitive structure, unnatural transitions, vague claims, or formulaic academic style.
Use only LOW / MODERATE / HIGH INDICATOR.
Important: Do not claim certainty.

## 11. Methodological Consistency Check
Check whether the research aim, research questions, methodology, sample, data analysis, results, and conclusions are logically aligned.
Identify inconsistencies.

## 12. References Requiring Crossref Validation
List references, DOI numbers, or cited works that should be checked in Crossref.

## 13. Final Editorial Recommendation
Give one recommendation:
- Acceptable
- Minor revision
- Major revision
- Reject

Briefly justify the recommendation.

Paper text:
{text[:MAX_TEXT_CHARS]}
"""


def generate_ai_report(text: str) -> str:
    prompt = create_structured_prompt(text)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.2
        }
    )

    return response["message"]["content"]


def build_reference_analysis(text: str) -> Dict[str, Any]:
    dois = extract_dois(text)
    duplicate_dois = find_duplicate_dois(dois)

    doi_validation = []
    valid_refs = []
    suspicious_refs = []
    possible_hallucinations = []

    for doi in dois:
        result = check_crossref_by_doi(doi)
        result["classification"] = classify_reference_status(result)

        doi_validation.append(result)

        if result["classification"] == "VALID":
            valid_refs.append(result)
        elif result["classification"] == "POSSIBLE_HALLUCINATION":
            possible_hallucinations.append(result)
        else:
            suspicious_refs.append(result)

    references = extract_reference_candidates(text, max_references=10)

    title_validation = []
    for reference in references:
        title_validation.append(check_crossref_by_title(reference))

    apa_references = [
        format_apa_reference(ref)
        for ref in valid_refs
    ]

    ieee_references = [
        format_ieee_reference(ref, i + 1)
        for i, ref in enumerate(valid_refs)
    ]

    return {
        "dois_found": dois,
        "duplicate_dois": duplicate_dois,
        "crossref_doi_validation": doi_validation,
        "valid_references": valid_refs,
        "suspicious_references": suspicious_refs,
        "possible_hallucinations": possible_hallucinations,
        "references_detected": references,
        "crossref_title_validation": title_validation,
        "apa_references": apa_references,
        "ieee_references": ieee_references
    }


@app.get("/analyze-text")
def analyze_text(text: str):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": f"""
Analyze the following text academically.

Return a clear report with:
1. Main idea
2. Academic relevance
3. Key concepts
4. Possible unsupported claims
5. Short conclusion

Text:
{text}
"""
            }
        ],
        options={
            "temperature": 0.2
        }
    )

    return {
        "input_text": text,
        "ai_analysis": response["message"]["content"]
    }


@app.get("/check-reference")
def check_reference(title: str):
    return check_crossref_by_title(title)


@app.post("/check-pdf-dois")
async def check_pdf_dois(file: UploadFile = File(...)):
    file_path = save_upload_file(file)

    try:
        await write_upload_to_disk(file, file_path)
        text = extract_pdf_text(file_path)
        dois = extract_dois(text)

        results = []
        for doi in dois:
            result = check_crossref_by_doi(doi)
            result["classification"] = classify_reference_status(result)
            results.append(result)

        return {
            "filename": file.filename,
            "total_dois_found": len(dois),
            "dois_found": dois,
            "duplicate_dois": find_duplicate_dois(dois),
            "crossref_doi_validation": results
        }

    finally:
        delete_temp_file(file_path)


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = save_upload_file(file)

    try:
        await write_upload_to_disk(file, file_path)
        text = extract_pdf_text(file_path)
        metadata = extract_basic_metadata(text)
        report = generate_ai_report(text)
        reference_analysis = build_reference_analysis(text)

        return {
            "filename": file.filename,
            "characters": len(text),
            "metadata": metadata,
            "structured_report": report,
            "reference_analysis": reference_analysis,
            "scopus_status": "Not checked - Scopus API key and institutional access required",
            "wos_status": "Not checked - Web of Science / Clarivate API access required",
            "note": "DOI validation is more reliable than title-based validation. Title-based validation and AI risk indicators should be manually verified."
        }

    finally:
        delete_temp_file(file_path)


@app.post("/upload-pdf-html", response_class=HTMLResponse)
async def upload_pdf_html(file: UploadFile = File(...)):
    file_path = save_upload_file(file)

    try:
        await write_upload_to_disk(file, file_path)
        text = extract_pdf_text(file_path)

        metadata = extract_basic_metadata(text)
        report = generate_ai_report(text)
        reference_analysis = build_reference_analysis(text)

        valid_refs = reference_analysis["valid_references"]
        suspicious_refs = reference_analysis["suspicious_references"]
        possible_hallucinations = reference_analysis["possible_hallucinations"]
        duplicate_dois = reference_analysis["duplicate_dois"]
        apa_references = reference_analysis["apa_references"]
        ieee_references = reference_analysis["ieee_references"]

        safe_report = html.escape(report)
        safe_filename = html.escape(file.filename or "Unknown file")
        safe_title = html.escape(metadata.get("estimated_title", "Unknown"))

        valid_html = "".join(
            f"<li><b>{html.escape(r.get('doi', ''))}</b> — "
            f"{html.escape(r.get('crossref_title', ''))} "
            f"({html.escape(str(r.get('year', 'n.d.')))})</li>"
            for r in valid_refs
        ) or "<li>No valid DOI references detected.</li>"

        suspicious_html = "".join(
            f"<li><b>{html.escape(r.get('doi', ''))}</b> — "
            f"{html.escape(r.get('status', 'UNKNOWN'))}: "
            f"{html.escape(r.get('message', 'Requires manual verification.'))}</li>"
            for r in suspicious_refs
        ) or "<li>No suspicious DOI references detected.</li>"

        hallucination_html = "".join(
            f"<li><b>{html.escape(r.get('doi', ''))}</b> — Possible hallucination or incomplete DOI.</li>"
            for r in possible_hallucinations
        ) or "<li>No possible DOI hallucinations detected.</li>"

        duplicates_html = "".join(
            f"<li>{html.escape(doi)}</li>"
            for doi in duplicate_dois
        ) or "<li>No duplicate DOI references detected.</li>"

        apa_html = "".join(
            f"<li>{html.escape(ref)}</li>"
            for ref in apa_references
        ) or "<li>No APA references generated.</li>"

        ieee_html = "".join(
            f"<li>{html.escape(ref)}</li>"
            for ref in ieee_references
        ) or "<li>No IEEE references generated.</li>"

        html_report = f"""
        <html>
        <head>
            <title>AI PDF Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                    background: #f7f9fb;
                    color: #222;
                }}
                .card {{
                    background: white;
                    padding: 25px;
                    margin-bottom: 20px;
                    border-radius: 12px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }}
                h1 {{ color: #1f4e79; }}
                h2 {{ color: #2f6f4e; }}
                pre {{
                    white-space: pre-wrap;
                    font-family: Arial, sans-serif;
                }}
                .valid {{ color: green; }}
                .warning {{ color: #b36b00; }}
                .danger {{ color: #b00020; }}
                .muted {{ color: #666; }}
            </style>
        </head>
        <body>
            <h1>AI Academic PDF Report</h1>

            <div class="card">
                <h2>NASLOV</h2>
                <p><b>File:</b> {safe_filename}</p>
                <p><b>Estimated title:</b> {safe_title}</p>
                <p><b>Characters extracted:</b> {len(text)}</p>
                <p><b>Total DOI found:</b> {len(reference_analysis["dois_found"])}</p>
            </div>

            <div class="card">
                <h2>SUMMARY / STRUCTURED REPORT</h2>
                <pre>{safe_report}</pre>
            </div>

            <div class="card">
                <h2 class="valid">VALID REFERENCES ✅</h2>
                <ul>{valid_html}</ul>
            </div>

            <div class="card">
                <h2 class="warning">SUSPICIOUS REFERENCES ⚠️</h2>
                <ul>{suspicious_html}</ul>
            </div>

            <div class="card">
                <h2 class="danger">POSSIBLE HALLUCINATIONS 🚨</h2>
                <ul>{hallucination_html}</ul>
                <p class="muted">References marked as NOT_FOUND, incomplete DOI, or mismatched metadata require manual editorial verification.</p>
            </div>

            <div class="card">
                <h2>DUPLICATE DOI REFERENCES</h2>
                <ul>{duplicates_html}</ul>
            </div>

            <div class="card">
                <h2>APA FORMATTED REFERENCES</h2>
                <ol>{apa_html}</ol>
            </div>

            <div class="card">
                <h2>IEEE FORMATTED REFERENCES</h2>
                <ol>{ieee_html}</ol>
            </div>

            <div class="card">
                <h2>SCOPUS / WEB OF SCIENCE STATUS</h2>
                <p>Scopus: Not checked — API key and institutional access required.</p>
                <p>Web of Science: Not checked — Clarivate API access required.</p>
            </div>
        </body>
        </html>
        """

        return Response(
    content=html_report,
    media_type="text/html",
    headers={
        "Content-Disposition": "attachment; filename=ai_pdf_report.html"
    }
)

    finally:
        delete_temp_file(file_path)
