from flask import Flask, request, render_template
from app.retriever import Retriever
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
retriever = Retriever()

import os

# Change to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

# Relative path to the model
model_path = "models/llama-2-7b-chat.Q2_K.gguf"

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

# Convert to absolute path
absolute_path = os.path.abspath(model_path)

# Print the absolute path
print(f"Absolute path: {absolute_path}")

# Verify the file exists
if not os.path.exists(absolute_path):
    raise FileNotFoundError(f"Model file not found at: {absolute_path}")
else:
    print("Model file found!")


# Define the data folder relative to the project root
data_folder = "data"

# Get the list of files in the data folder
documents = [os.path.join(data_folder, file_name)
             for file_name in os.listdir(data_folder)
             if file_name.endswith(".pdf") or file_name.endswith(".txt") or file_name.endswith(".docx")]


absolute_documents = [os.path.abspath(doc) for doc in documents]

# Print the list of documents
print("Documents to process:", absolute_documents)

# Load embeddings (generate them first if they don't exist)
retriever.add_documents(absolute_documents)
retriever.save()

# Load LLaMA 2
llm = LlamaCpp(
    model_path=model_path,  # Path to your LLaMA 2 model
    temperature=0.7,
    max_tokens=150,
    n_ctx=2048,
)


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


@app.route("/", methods=["GET", "POST"])
def home():
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
