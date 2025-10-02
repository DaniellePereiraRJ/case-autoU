
"""Email Classifier & Auto-Responder - Flask demo app (single-file)
Save as app.py and run: python app.py
If you want OpenAI-enhanced replies, set OPENAI_API_KEY env var.
"""
from flask import Flask, request, render_template_string, send_file
import os, io, re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pdfplumber
import openai

# Ensure NLTK data is available (first-run may download into user's NLTK path)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except Exception:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

STOPWORDS = set(stopwords.words('english'))
LEM = WordNetLemmatizer()

TRAIN_SAMPLES = [
    ("Hi, can I get an update on ticket #123? The client is asking for ETA.", "Produtivo"),
    ("Please share the contract and signed agreement for client ABC.", "Produtivo"),
    ("Happy holidays and best wishes to the whole team!", "Improdutivo"),
    ("Thanks for your help earlier, much appreciated.", "Improdutivo"),
    ("There is a discrepancy in the latest statement, please investigate.", "Produtivo"),
    ("Congratulations on the new year everyone!", "Improdutivo"),
    ("Attached is the bank reconciliation file. Please confirm receipt.", "Produtivo"),
    ("Just saying hello and hope you're well.", "Improdutivo"),
]

def preprocess_text(text: str) -> str:
    text = (text or '').lower()
    text = re.sub(r"\s+", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [LEM.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def train_classifier(samples):
    texts = [preprocess_text(s[0]) for s in samples]
    labels = [s[1] for s in samples]
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    clf = LogisticRegression(max_iter=1000)
    pipeline = make_pipeline(vect, clf)
    pipeline.fit(texts, labels)
    return pipeline

MODEL = train_classifier(TRAIN_SAMPLES)

def suggest_reply(category: str, original_text: str) -> str:
    if category == 'Produtivo':
        base = (
            "Olá, obrigado pelo contato. Recebemos sua mensagem e estamos analisando sua solicitação. "
            "Por favor, informe qualquer informação adicional relevante (números de pedido, anexos ou prazos)."
        )
    else:
        base = "Olá! Agradecemos sua mensagem. Ela não requer ação imediata, mas ficamos à disposição caso precise de algo."

    api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_KEY')
    if api_key:
        try:
            openai.api_key = api_key
            prompt = (
                f"You are an assistant for a financial services company. The incoming email is:\n\n{original_text}\n\n"
                f"The email was classified as {category}. Write a polite, professional reply in Portuguese, 2-4 sentences, including a brief next step if relevant."
            )
            resp = openai.ChatCompletion.create(
                model='gpt-4o-mini',
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
                max_tokens=200,
            )
            gen = resp['choices'][0]['message']['content'].strip()
            return gen
        except Exception as e:
            print('OpenAI call failed:', e)
            return base + "\n\n(Note: generation via OpenAI failed; returned template used.)"
    else:
        return base

def extract_text_from_pdf(file_stream) -> str:
    text = ""
    try:
        file_stream.seek(0)
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ''
                text += '\n' + page_text
    except Exception as e:
        print('PDF parsing error:', e)
    return text

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html lang="pt-br">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Email Classifier — Demo</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { background: linear-gradient(120deg,#f8fafc,#eef2ff); }
      .card { box-shadow: 0 6px 18px rgba(15,23,42,0.08); border-radius:12px }
      textarea { min-height:160px }
      .result-badge { font-weight:600 }
    </style>
  </head>
  <body>
    <div class="container py-5">
      <div class="row justify-content-center">
        <div class="col-md-9">
          <div class="card p-4">
            <h3 class="mb-1">Classificador de Emails — Demo</h3>
            <p class="text-muted">Cole o texto do email ou envie um arquivo (.txt ou .pdf). O sistema irá classificar e sugerir uma resposta automática.</p>

            <form id="emailForm" method="post" action="/process" enctype="multipart/form-data">
              <div class="mb-3">
                <label class="form-label">Colar texto do email</label>
                <textarea name="email_text" class="form-control" placeholder="Cole aqui o corpo do email..."></textarea>
              </div>
              <div class="mb-3">
                <label class="form-label">Ou faça upload do arquivo</label>
                <input class="form-control" type="file" name="email_file" accept=".txt,.pdf">
              </div>
              <div class="d-flex gap-2">
                <button class="btn btn-primary" type="submit">Processar</button>
                <button class="btn btn-outline-secondary" type="button" onclick="fillExamples()">Exemplo</button>
                <a class="btn btn-link text-muted" href="/download_sample">Baixar amostra</a>
              </div>
            </form>

            {% if result %}
            <hr>
            <h5>Resultado</h5>
            <p>Categoria: <span class="badge bg-{{ 'success' if result.category=='Produtivo' else 'secondary' }} result-badge">{{ result.category }}</span></p>
            <h6>Resposta sugerida</h6>
            <div class="border rounded p-3 bg-white">
              <pre style="white-space:pre-wrap;">{{ result.suggested_reply }}</pre>
            </div>
            <h6 class="mt-3">Trecho do email (pré-processado)</h6>
            <div class="small text-muted">{{ result.preview }}</div>
            {% endif %}

          </div>
          <div class="text-center mt-3 small text-muted">Feito para desafio — adaptável para produção.</div>
        </div>
      </div>
    </div>

    <script>
      function fillExamples(){
        const area = document.querySelector('textarea[name=email_text]');
        area.value = "Olá, gostaria de saber o status do protocolo 2024-567. Já faz 10 dias e não tivemos atualização.\nAtenciosamente, Cliente";
      }
    </script>
  </body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML)

@app.route('/download_sample')
def download_sample():
    sample = ("Assunto: Solicitação de atualização\n\nOlá,\n\nGostaria de saber o status do protocolo 2024-567. Por favor, envie atualização.\n\nAtenciosamente,\nCliente")
    return send_file(io.BytesIO(sample.encode('utf-8')), mimetype='text/plain', as_attachment=True, download_name='sample_email.txt')

@app.route('/process', methods=['POST'])
def process():
    text_input = request.form.get('email_text', '').strip()
    uploaded = request.files.get('email_file')
    original_text = ''

    if uploaded and uploaded.filename != '':
        fname = uploaded.filename.lower()
        if fname.endswith('.txt'):
            original_text = uploaded.stream.read().decode('utf-8', errors='ignore')
        elif fname.endswith('.pdf'):
            uploaded.stream.seek(0)
            original_text = extract_text_from_pdf(uploaded.stream)
        else:
            original_text = uploaded.stream.read().decode('utf-8', errors='ignore')
    else:
        original_text = text_input

    if not original_text:
        return render_template_string(INDEX_HTML, result=None, error='Nenhum texto enviado')

    preview = preprocess_text(original_text)[:800]
    cat = MODEL.predict([preprocess_text(original_text)])[0]
    suggested = suggest_reply(cat, original_text)

    result = {
        'category': cat,
        'suggested_reply': suggested,
        'preview': preview
    }

    return render_template_string(INDEX_HTML, result=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000) or 5000)
    app.run(host='0.0.0.0', port=port, debug=True)
