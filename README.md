## AIM:

To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

## PROBLEM STATEMENT:

To develop a chatbot that allows users to upload a PDF document and ask questions based on its content. The system retrieves relevant information and generates accurate answers using LangChain and a Large Language Model.

## DESIGN STEPS:

#### STEP 1: PDF Upload and Text Extraction
- Install the required Python libraries.
- Upload a PDF document using the Jupyter Notebook interface.
- Extract text from the PDF using PyPDFLoader.

#### STEP 2: Document Processing
- Split the extracted text into smaller chunks.
- Generate embeddings using Hugging Face.
- Store the embeddings in a FAISS vector database.

#### STEP 3: Question Answering
- Accept questions from the user.
- Retrieve relevant document chunks using similarity search.
- Generate answers using the Groq LLM through LangChain.
- Display answers along with source page references.

## PROGRAM:

```python
%pip install -q langchain langchain-community langchain-groq langchain-huggingface langchain-text-splitters faiss-cpu pypdf sentence-transformers ipywidgets
```

```python
import os
import torch
import ipywidgets as widgets
from IPython.display import display
from getpass import getpass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# API Configuration
os.environ["GROQ_API_KEY"] = getpass("Enter Groq API Key: ")

# Upload PDF
upload = widgets.FileUpload(accept=".pdf", multiple=False)
display(upload)

# After uploading, run this cell
file_info = upload.value

if isinstance(file_info, dict):
    file_name = next(iter(file_info))
    file_content = file_info[file_name]["content"]
else:
    file_name = file_info[0]["name"]
    file_content = file_info[0]["content"]

with open(file_name, "wb") as f:
    f.write(file_content)

# Load PDF
loader = PyPDFLoader(file_name)
pages = loader.load()

# Split Text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
chunks = splitter.split_documents(pages)

# Generate Embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# Create FAISS Vector Store
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

# Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# Create Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the given context.
If the answer is unavailable, say:
"The answer is not available in the provided document."

Context:
{context}

Question: {question}
""")

def ask_pdf(question):
    docs = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in docs)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    print("Answer:", answer)
    print("Source Pages:",
          sorted(set(doc.metadata["page"] + 1 for doc in docs)))

# Ask Questions
while True:
    question = input("\nAsk a question (or type exit): ")

    if question.lower() == "exit":
        break

    ask_pdf(question)
```

## OUTPUT:

<img width="1352" height="797" alt="image" src="https://github.com/user-attachments/assets/cb2870bd-d926-460c-be3a-a893e3968a69" />
<img width="1237" height="732" alt="image" src="https://github.com/user-attachments/assets/82ec0053-dddd-440c-8ed5-1857b31da6d1" />


## RESULT:

Thus, the PDF-based Question-Answering Chatbot was successfully designed and implemented using LangChain. The system processed the uploaded PDF, answered user queries, and provided relevant source page references.
