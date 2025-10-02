📧 Email Classifier Demo

Aplicação web simples que utiliza Inteligência Artificial para classificar emails em Produtivos ou Improdutivos, além de sugerir uma resposta automática adequada.

O projeto foi desenvolvido em Python + Flask, com técnicas de Processamento de Linguagem Natural (NLP) para pré-processamento de texto e integração opcional com OpenAI GPT para gerar respostas mais avançadas.

🎯 Objetivo

Automatizar a leitura e classificação de emails.

Identificar se o email exige ação (Produtivo) ou não (Improdutivo).

Sugerir respostas automáticas para cada caso.

Reduzir o tempo da equipe em tarefas manuais repetitivas.

🖥️ Interface Web

A interface web permite:

Upload de arquivos .txt ou .pdf contendo emails.

Inserção manual de texto de emails.

Visualização da categoria atribuída (Produtivo / Improdutivo).

Exibição da resposta automática sugerida.

⚙️ Tecnologias Utilizadas

Python 3.9+

Flask (servidor web)

NLTK (pré-processamento NLP)

scikit-learn (classificação simples via TF-IDF + Logistic Regression)

pdfplumber (extração de texto de PDFs)

OpenAI API (opcional) para respostas automáticas avançadas

Bootstrap para estilização simples do frontend
