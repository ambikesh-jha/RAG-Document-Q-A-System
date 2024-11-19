from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from utils.pdf_parser import extract_text_from_pdf
from utils.retriever import retrieve_answer
from utils.llm_integration import ask_external_llm

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads/'

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if not file.filename.endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        return jsonify({"message": "File uploaded successfully!", "filename": file.filename})
    except Exception as e:
        return jsonify({"error": f"Failed to upload the file. Error: {str(e)}"}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question')
        filename = data.get('filename')

        if not question or not filename:
            return jsonify({"answer": "Invalid request. Please provide both question and filename."}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Extract text from PDF
        document_text = extract_text_from_pdf(filepath)

        # Retrieve answer from PDF or fallback to external LLM
        answer = retrieve_answer(question, document_text)
        if not answer:
            answer = ask_external_llm(question)

        return jsonify({"answer": answer})  # Ensure this is JSON format
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
