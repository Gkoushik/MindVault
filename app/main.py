from flask import Flask, request, render_template, redirect, url_for
from app.retriever import Retriever
from werkzeug.utils import secure_filename
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

retriever = Retriever()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

model_path = "models/llama-2-7b-chat.Q2_K.gguf"

# Load LLaMA 2
llm = LlamaCpp(
    model_path=model_path,  # Path to your LLaMA 2 model
    temperature=0.7,
    max_tokens=150,
    n_ctx=2048,
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def generate_response(query, context):
    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template="Based on the following context:\n{context}\n\nAnswer the question: {query}",
    )

    # Create an LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Generate the response
    response = llm_chain.run(context=context, query=query)
    return response.strip()


# List to store uploaded file names
uploaded_files = []


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Handle file upload
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(filename)
            retriever.add_documents([file_path])
            return redirect(url_for('home'))

    return render_template("index.html", uploaded_files=uploaded_files)


@app.route("/search", methods=["POST"])
def search():
    if request.method == "POST":
        query = request.form["query"]
        # Retrieve relevant documents
        relevant_docs = retriever.search(query)
        context = "\n".join(relevant_docs)
        print(context, "context:")
        # Generate a response using the local LLM
        response = generate_response(query, context)
        return render_template("index.html", query=query, response=response, context=context)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
