import PyPDF2.errors
from flask import Blueprint, render_template, flash, redirect
from app.pdf.forms import UploadPDFForm, PDFQAForm
from app import db_client, app
import os
import PyPDF2
from werkzeug.utils import secure_filename
from flask import request
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import  HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
from bson import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)
from langchain.prompts import load_prompt
from langchain.chains.question_answering import load_qa_chain
import re



# Load Google Gen AI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create a blueprint
upload_pdf_route = Blueprint('pdf', __name__, template_folder='templates')

# VECTOR
VECTOR_DB_PATH = os.path.abspath(os.path.join(app.root_path, 'static/vector/pdf'))

# File Upload
UPLOAD_FOLDER = os.path.abspath(os.path.join(app.root_path, 'static/tempfs'))

ALLOWED_EXTENSIONS = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_pdf(file_path):
    try:
        # Open the PDF file
        with open(file_path, 'rb') as pdf_file:
            # Create a PdfFileReader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Check if the file has at least one page
            if len(pdf_reader.pages) > 0:
                return True
            else:
                return False
    except PyPDF2.errors.PdfStreamError as e:
        # If the file cannot be read as a PDF
        return False
    except FileNotFoundError as e:
        # If the file is not found
        return False

def get_pdf_file_size(file_path) -> str:

    # Get the file size in bytes
    size_bytes = os.path.getsize(file_path)
    # Determine the appropriate unit (KB or MB) based on file size
    if size_bytes < 1024:
        file_size = f'{size_bytes} bytes'
    elif size_bytes < 1024 * 1024:
        file_size = f'{size_bytes / 1024:.2f} KB'
    else:
        file_size = f'{size_bytes / (1024 * 1024):.2f} MB'

    return file_size


@upload_pdf_route.route('/upload_pdf', methods=['GET', 'POST'])
def upload_pdf():
    form = UploadPDFForm()
    # Create Connection to MongoDB database
    db = db_client.get_database('documents')
    # Get the collection
    collection = db.get_collection('pdf')
    # Check if the collection is empty
    if collection.count_documents({}) == 0:
        pdf_doc_list = []
    else:
        # Get all documents in the collection
        pdf_doc_list = collection.find({})
        pdf_doc_list = [doc for doc in pdf_doc_list]
        
    if form.validate_on_submit():
        # Get the file from the form
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if file and allowed_file(file.filename):     
            # Save the file to the static/tempfs folder
            file.save(filepath)
            
            # Check if the file is a valid PDF
            if not is_valid_pdf(filepath):
                flash(f'Invalid file type. Please upload a PDF file.', 'danger')
                return redirect('/upload_pdf')
            
            # Get PDF file size
            file_size = get_pdf_file_size(filepath)

            # Read and split the PDF file, Wait untill the file is saved
            chunks = read_split_pdf_chunk(filepath)
            if chunks == False:
                flash(f'Error splitting PDF file. Please try again.', 'danger')
                return redirect('/upload_pdf')
            
            # Embed chunks
            embedding_result = embed_chunks(chunks, filename)
            if not embedding_result:
                flash(f'Error embedding PDF chunks. Please try again.', 'danger')
                return redirect('/upload_pdf')
            
            # Copy the PDF file to Vector DB path
            shutil.copy(filepath, embedding_result[0])

            # Set PDF path from static folder
            pdf_path = f"/vector/pdf/{embedding_result[1]}/{filename}"

            # Delete the uploaded pdf file from the static/tempfs folder
            os.remove(filepath)

            # Save the file to the database
            collection.insert_one({'filename': filename, 
                                   'file_size': file_size, 
                                   'vector_db_name': embedding_result[1], 
                                   'vector_db_path': embedding_result[0], 
                                   'pdf_path': pdf_path})
            flash(f'{filename} uploaded successfully.', 'success')
            return redirect('/upload_pdf')
        else:
            flash(f'Invalid file type. Please upload a PDF file.', 'danger')
            return redirect('/upload_pdf')

    return render_template('pdf/upload_pdf.html', title="DocuSense AI: Upload PDF", pdf_doc_list=pdf_doc_list, form=form)


# Load the PDF file and split it into smaller chunks
def read_split_pdf_chunk(pdf_file_path):
    try:
        # pdf_text = ""
        # pdf_reader = PdfReader(pdf_file_path)
        # for page in pdf_reader.pages:
        #     pdf_text += page.extract_text()
        # Load the PDF file
        loader = PyPDFLoader(pdf_file_path)
        # Load the PDF file
        pdf_text = loader.load()

        # Split the PDF file into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        chunks = text_splitter.split_documents(pdf_text)
        # Pre-Process chunks
        for i, chunk in enumerate(chunks):
            chunks[i].page_content = preprocess_text(chunk.page_content)
        return chunks
    except Exception as e:
        return False
    
# Pre-Process Chunk Text
def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Retrieve embedding function from code env resources
def embed_chunks(chunks, filename):
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # Create an embeddings model
        embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
        # Create a vector store from the PDF chunks
        vector_db_path = os.path.join(VECTOR_DB_PATH, filename+"_emb")
        # Save Chunks in Chroma DB
        Chroma.from_documents(chunks, embedding=embedding_function, persist_directory=vector_db_path, collection_name=filename+"_emb", collection_metadata={"source":f"{filename}"+"_emb"})
        return [vector_db_path,  filename+"_emb"]
    except Exception as e:
        return False
    # try:
    #     # Filename, remove .pdf from filename
    #     filename = filename[:-4]
    #     # Create an embeddings model
    #     # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #     # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    #     embedding_function = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #     # vector_store =  FAISS.from_texts(chunks, embedding=embeddings)
    #     # Create a vector store from the PDF chunks
    #     vector_db_path = os.path.join(VECTOR_DB_PATH, filename+"_emb")
    #     # Create Vector db path folder
    #     #os.makedirs(vector_db_path,exist_ok=True)
    #     # Save the vector store to the local file system
    #     # vector_store.save_local(vector_db_path)
    #     # Chroma.from_documents(chunks,embedding=embeddings,persist_directory=vector_db_path, collection_name=filename+"_emb", collection_metadata={"source":f"{filename}"+"_emb"})
    #     Chroma.from_documents(chunks, embedding=embedding_function, persist_directory=vector_db_path, collection_name=filename+"_emb", collection_metadata={"source":f"{filename}"+"_emb"})
        
    #     return vector_db_path,  filename+"_emb"
    # except Exception as e:
    #     print(e)
    #     return False

# Delete PDF from database
@upload_pdf_route.route('/delete_pdf/<string:docid>', methods=['GET', 'POST'])
def delete_pdf(docid):
    # Create Connection to MongoDB database
    db = db_client.get_database('documents')
    # Get the collection
    collection = db.get_collection('pdf')
    # Get the document with the given id
    doc = collection.find_one({"_id": ObjectId(docid)})
    # Get the vector database path from the document
    vector_db_path = doc['vector_db_path']
    # Delete the vector database from the static/vector/pdf folder
    vector_db_path = os.path.join(VECTOR_DB_PATH, vector_db_path)
    shutil.rmtree(vector_db_path)
    # Delete the document from the collection
    collection.delete_one({"_id": ObjectId(docid)})
    flash(f'PDF File {doc["filename"]} deleted successfully.', 'success')
    return redirect('/upload_pdf')

# Create conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Create Langchain Question Answer Chain
def get_langchain_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Query PDF with document ID
@upload_pdf_route.route('/query_pdf/<string:docid>', methods=['GET', 'POST'])
def query_pdf(docid):
    response = {}
    # Create Connection to MongoDB database
    db = db_client.get_database('documents')
    # Get the collection
    collection = db.get_collection('pdf')
    # Get the document with the given id
    doc = collection.find_one({"_id": ObjectId(docid)})
    # Get the PDF file path from the document
    pdf_path = doc['pdf_path']
    
    form = PDFQAForm()

    if form.validate_on_submit():
        # Get the question from the form
        user_question = form.question.data
        # Embedding Model
        embedding_function = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory=doc['vector_db_path'], embedding_function=embedding_function, collection_name=doc['vector_db_name'])
        docs = db.similarity_search(user_question)
        chain = get_langchain_chain()
        response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)
        print(response)
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # new_db = FAISS.load_local(r"D:\\DocuSenseAI\\app\\static\\vector\\pdf\\bs2023_24_emb", embeddings, allow_dangerous_deserialization=True)
        # docs = new_db.similarity_search(user_question)
        # chain = get_conversational_chain()
        # response = chain({"input_documents":docs,"question": user_question},return_only_outputs=True)
    return render_template('pdf/query_pdf.html', title="DocuSense AI: Query PDF", pdf_path=pdf_path, form=form, answer=response)
