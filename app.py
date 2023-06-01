from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import requests
import io
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

load_dotenv()
app = Flask(__name__)
CORS(app)

knowledge_base = None
current_pdf_filename = None


@app.route('/process', methods=['POST'])
def process_pdf():
    global current_pdf_filename
    # If a new file is being processed, delete the old one
    if 'pdf' in request.files:
        pdf = request.files['pdf']
        current_pdf_filename = secure_filename(pdf.filename)
        pdf_filepath = os.path.join('pdf_files', current_pdf_filename)
        pdf.save(pdf_filepath)
    # If not, check if a URL was sent
    elif 'url' in request.json:
        pdf_url = request.json['url']
        response = requests.get(pdf_url)
        response.raise_for_status()
        current_pdf_filename = pdf_url.split('/')[-1]
        pdf_filepath = os.path.join('pdf_files', current_pdf_filename)
        with open(pdf_filepath, 'wb') as out_file:
            out_file.write(response.content)
    else:
        return jsonify({'message': 'No PDF file or URL provided'}), 400

    with open(pdf_filepath, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

    with open('last_pdf.txt', 'w') as file:
        file.write(current_pdf_filename)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    global knowledge_base
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    response = jsonify({'message': 'PDF processed successfully'})
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route('/query', methods=['POST'])
def answer_question():
    if knowledge_base is None:
        return jsonify({"answer": "aucun pdf n'est chargé"})

    user_question = request.json['question']
    docs = knowledge_base.similarity_search(user_question)

    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)

    response_with_header = jsonify({'answer': response})
    response_with_header.headers.add('Access-Control-Allow-Origin', '*')

    return response_with_header


@app.route('/current_title', methods=['GET'])
def get_current_title():
    if current_pdf_filename is not None:
        return jsonify({'title': current_pdf_filename})
    else:
        return jsonify({'message': 'No PDF file has been processed yet'}), 400


if __name__ == '__main__':
    if not os.path.exists('pdf_files'):
        os.makedirs('pdf_files')

    if os.path.exists('last_pdf.txt'):
        with open('last_pdf.txt', 'r') as file:
            current_pdf_filename = file.read()
            pdf_filepath = os.path.join('pdf_files', current_pdf_filename)
            if os.path.exists(pdf_filepath):
                # Process the file to load knowledge_base
                with open(pdf_filepath, 'rb') as pdf_file:
                    pdf_reader = PdfReader(pdf_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)

                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embeddings)
            else:
                print(f"File {current_pdf_filename} does not exist")

    app.run(port=8000, debug=True)
