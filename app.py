import os
import PyPDF2
import spacy
import pandas as pd
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

# Create folders if they don't exist
os.makedirs("resumes", exist_ok=True)
os.makedirs("ranked_reports", exist_ok=True)

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def rank_resumes(resume_texts, job_description):
    processed_resumes = [preprocess(text) for text in resume_texts]
    processed_jd = preprocess(job_description)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([processed_jd] + processed_resumes)

    scores = (vectors[1:] * vectors[0].T).toarray()
    return [score[0] for score in scores]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_desc = request.form["job_description"]
        uploaded_files = request.files.getlist("resumes")

        resume_texts = []
        filenames = []

        for file in uploaded_files:
            filename = file.filename
            filepath = os.path.join("resumes", filename)
            file.save(filepath)
            text = extract_text_from_pdf(filepath)
            resume_texts.append(text)
            filenames.append(filename)

        scores = rank_resumes(resume_texts, job_desc)
        ranked = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)

        df = pd.DataFrame(ranked, columns=["Filename", "Score"])
        output_path = "ranked_reports/report.csv"
        df.to_csv(output_path, index=False)

        return render_template("result.html", ranked=ranked, download_link=output_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
