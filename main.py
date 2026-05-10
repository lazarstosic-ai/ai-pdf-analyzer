from fastapi import FastAPI
import ollama

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Local AI Assistant Running"}

@app.get("/analyze-text")
def analyze_text(text: str):

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": f"Analyze this text academically: {text}"
            }
        ]
    )

    return {
        "ai_analysis": response["message"]["content"]
    }
from fastapi import UploadFile, File
from pypdf import PdfReader

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    with open(file.filename, "wb") as f:
        f.write(await file.read())

    reader = PdfReader(file.filename)

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": f"""
                Analyze this academic paper.

                Provide:
                1. Summary
                2. Main topic
                3. Methodology
                4. Key findings
                5. Possible limitations

                Paper:
                {text[:12000]}
                """
            }
        ]
    )

    return {
        "filename": file.filename,
        "characters": len(text),
        "ai_analysis": response["message"]["content"]
    }```bash id="3z7uk2"
