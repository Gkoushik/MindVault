from flask import Flask, request, render_template
import os
from dotenv import load_dotenv
from app.retriever import Retriever

load_dotenv()

app = Flask(__name__)
retriever = Retriever()

# Load documents (you can replace this with your own data)
with open("data/documents.txt", "r") as f:
    documents = f.readlines()

# Load embeddings (generate them first if they don't exist)
if not os.path.exists("faiss_index.index"):
    retriever.add_documents(documents)
    retriever.save()
retriever.load()

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":
        query = request.form["query"]
        relevant_indices = retriever.search(query)
        relevant_docs = [documents[idx].strip() for idx in relevant_indices]
        return render_template("index.html", query=query, results=relevant_docs)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
