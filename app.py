from flask import Flask, render_template, request, jsonify
import os
from utils.file_parser import extract_text
from utils.retriever import index , create_faiss_index, query_faiss_index
from utils.llm_integration import ask_external_llm

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads/'

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index_html():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not (file.filename.endswith('.pdf') or file.filename.endswith(('.doc', '.docx'))):
            return jsonify({"error": "Only PDF or Word files are allowed"}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract text and update FAISS index
        document_text = extract_text(filepath)
        if document_text:
            create_faiss_index(document_text, index, file.filename)

        return jsonify({"message": "File uploaded successfully!", "filename": file.filename})
    except Exception as e:
        return jsonify({"error": f"Failed to upload the file. Error: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question')

        if not question:
            return jsonify({"answer": "Invalid request. Please provide a question."}), 400

        # Query FAISS index
        result = query_faiss_index(question, index)
        if result:
            return jsonify({"answer": result})

        # Fallback to external LLM
        answer = ask_external_llm(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
