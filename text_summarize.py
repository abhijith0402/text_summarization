from flask import Flask, render_template, request, jsonify, send_file
import math
import nltk
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
from googletrans import Translator
from collections import Counter
from gensim import corpora, models
from langdetect import detect
from PyPDF2 import PdfReader
import docx
from fpdf import FPDF
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

nltk.download('wordnet')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()
stopWords = set(stopwords.words("english"))
translator = Translator()
summarizer = pipeline("summarization")
sentiment_pipeline = pipeline("sentiment-analysis")

app = Flask(__name__)

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    return " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())

def detect_language(text):
    return detect(text)

def extract_keywords(text, num_keywords=5):
    words = [word.text.lower() for word in nlp(text) if word.is_alpha and word.text.lower() not in stopWords]
    return [word for word, _ in Counter(words).most_common(num_keywords)]

def extract_topics(text, num_topics=3):
    tokens = [[word.text.lower() for word in nlp(sent.text) if word.is_alpha and word.text.lower() not in stopWords] 
              for sent in nlp(text).sents]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    return [topic[1] for topic in lda_model.print_topics(num_words=5)]

def extract_entities(text):
    doc = nlp(text)
    return {ent.text: ent.label_ for ent in doc.ents}

def transformer_summarize(text, length_option="medium"):
    if length_option == "short":
        min_length, max_length = 30, 100
    elif length_option == "long":
        min_length, max_length = 200, 500
    else: 
        min_length, max_length = 100, 200

    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

def analyze_sentiment(text):
    return sentiment_pipeline(text)[0]['label']

def generate_pdf(summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(190, 10, summary)
    pdf.output("summary.pdf")
    return "summary.pdf"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/summarization')
def summarization():
    return render_template('summarization.html')

@app.route('/summarize', methods=['POST'])
def summarize_file():
    file = request.files['file']
    summary_length = request.form.get("summary_length", "medium")
    show_original = request.form.get("show_original", "off")
    file_ext = file.filename.split('.')[-1]
    
    if file_ext.lower() == 'pdf':
        text = extract_text_from_pdf(file)
    elif file_ext.lower() in ['docx', 'doc']:
        text = extract_text_from_docx(file)
    else:
        text = file.read().decode('utf-8')
    
    detected_lang = detect_language(text)
    if detected_lang != 'en':
        text = translator.translate(text, dest='en').text
    
    summary = transformer_summarize(text, summary_length)
    sentiment = analyze_sentiment(summary)
    keywords = extract_keywords(text)
    topics = extract_topics(text)
    entities = extract_entities(text)
    
    response = {
        'summary': summary,
        'original_text': text if show_original == "on" else "",
        'original_word_count': len(text.split()),
        'summary_word_count': len(summary.split()),
        'summary_sentiment': sentiment,
        'keywords': keywords,
        'topics': topics,
        'named_entities': entities,
        'language_detected': detected_lang
    }
    
    return jsonify(response)

@app.route('/summarize_url', methods=['POST'])
def summarize_url():
    url = request.form.get('url')
    summary_length = request.form.get("summary_length", "medium")
    show_original = request.form.get("show_original", "off")
    
    if not url:
        return jsonify({'error': 'URL is required.'}), 400
    
    text = extract_text_from_url(url)
    detected_lang = detect_language(text)
    if detected_lang != 'en':
        text = translator.translate(text, dest='en').text
    
    summary = transformer_summarize(text, summary_length)
    sentiment = analyze_sentiment(summary)
    keywords = extract_keywords(text)
    topics = extract_topics(text)
    entities = extract_entities(text)
    
    response = {
        'summary': summary,
        'original_text': text if show_original == "on" else "",
        'original_word_count': len(text.split()),
        'summary_word_count': len(summary.split()),
        'summary_sentiment': sentiment,
        'keywords': keywords,
        'topics': topics,
        'named_entities': entities,
        'language_detected': detected_lang
    }
    
    return jsonify(response)

@app.route('/download_summary', methods=['POST'])
def download_summary():
    summary_text = request.json.get('summary', '')
    pdf_filename = generate_pdf(summary_text)
    return send_file(pdf_filename, as_attachment=True)

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    data = request.json
    text = data.get('text', '')
    summary = transformer_summarize(text)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
