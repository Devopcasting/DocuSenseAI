from flask import Blueprint, render_template, flash, redirect, request
from app.excel.forms import UploadExcelFForm, ExcelQAForm
from app import db_client, app
import os
from werkzeug.utils import secure_filename
import pandas as pd
from bson import ObjectId
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader
import re
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Chroma
import shutil
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import DataFrameLoader
import datetime

# Load Google Gen AI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# File Upload
UPLOAD_FOLDER = os.path.abspath(os.path.join(app.root_path, 'static/tempfs'))

# VECTOR
VECTOR_DB_PATH = os.path.abspath(os.path.join(app.root_path, 'static/vector/excel'))

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create a Blueprint object
upload_excel_route = Blueprint('excel', __name__, template_folder='templates')

# Get the size of uploaded file
def get_file_size(file_path):
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

def is_valid_file(file_path):
    file_extension = file_path.lower().split('.')[-1]
    
    if file_extension == 'xlsx' or file_extension == 'xls':
        # Check if it's a valid Excel file and not blank
        return is_valid_excel(file_path)
    elif file_extension == 'csv':
        # Check if it's a valid CSV file and not blank
        return is_valid_csv(file_path)
    else:
        # Unsupported file format
        return False
    
def is_valid_csv(file_path):
    try:
        # Create a Pandas DataFrame object
        df = pd.read_csv(file_path)
        # Check if the file has at least one row
        if len(df) > 0:
            return True
        else:
            return False
    except Exception as e:
        return False
    
def is_valid_excel(file_path):
    try:
        # Create a Pandas DataFrame object
        df = pd.read_excel(file_path)
        # Check if the file has at least one row
        if len(df) > 0:
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False
    
@upload_excel_route.route('/upload_excel', methods=['GET', 'POST'])
def upload_excel():
    form = UploadExcelFForm()
    # Create Connection to MongoDB database
    db = db_client.get_database('documents')
    # Get the collection
    collection = db.get_collection('excel')
    # Check if the collection is empty
    if collection.count_documents({}) == 0:
        excel_doc_list = []
    else:
        # Get all documents in the collection
        excel_doc_list = collection.find({})
        excel_doc_list = [doc for doc in excel_doc_list]
    
    if form.validate_on_submit():
        # Get the file from the form
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if file:     
            # Save the file to the static/tempfs folder
            file.save(filepath)
            # Check if the file is a valid Excel file
            if not is_valid_file(filepath):
                flash('Invalid file type. Please upload an Excel file.', 'danger')
                return redirect('/upload_excel')
            
            # Get the file size
            file_size = get_file_size(filepath)
            
            """"Load Excel file and Split into chunks"""
            chunks = load_excel(filepath)
            if not chunks:
                flash('Error loading Excel file.', 'danger')
                return redirect('/upload_excel')

            """"Embed chunks in Chroma DB"""
            embedding_result = embed_chunks(chunks, filename)
            if not embedding_result:
                flash('Error embedding Excel chunks.', 'danger')
                return redirect('/upload_excel')
            
            # Copy the Excel file to Vector DB path
            shutil.copy(filepath, embedding_result[0])

            # Set Excel path from static folder
            excel_path = f"/vector/excel/{embedding_result[1]}/{filename}"

            # Delete the uploaded Excel file from the static/tempfs folder
            os.remove(filepath)
            # Add the document to the collection
            collection.insert_one({"filename": filename, "file_size": file_size, "excel_path": excel_path, "vector_db_path": embedding_result[0], "vector_db_name": embedding_result[1]})
            
            flash(f'{filename} uploaded successfully.', 'success')
            return redirect('/upload_excel')
    
    return render_template('excel/upload_excel.html', title="DocuSense AI: Upload Excel", form=form, excel_doc_list=excel_doc_list)

# Load the Excel file
def load_excel(filepath):
    df = pd.read_excel(filepath)
    loader = DataFrameLoader(df)
    excel_text = loader.load()
    print(excel_text)
    # try:
    #     excel_loader = UnstructuredExcelLoader(filepath)
    #     excel_text = excel_loader.load()
    #     # Split into chunks
    #     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
    #     chunks = text_splitter.split_documents(excel_text)
    #     # Pre-Process the chunks
    #     for i, chunk in enumerate(chunks):
    #         chunks[i].page_content = preprocess_text(chunk.page_content)
    #     return chunks
    # except Exception as e:
    #     print(e)
    #     return False

# Pre-Process Chunk Text
def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create Chunks of Excel documents and store embedded in Chroma DB
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

# Query Excel
@upload_excel_route.route('/query_excel/<string:docid>', methods=['GET', 'POST'])
def query_excel(docid):
    form = ExcelQAForm()
    response = {}
    # Create Connection to MongoDB database
    db = db_client.get_database('documents')
    # Get the collection
    collection = db.get_collection('excel')
    # Get the document by ID
    doc = collection.find_one({"_id": ObjectId(docid)})
    # Get the filename
    filename = doc['filename']
    # Get the excel file path
    excel_path = doc['excel_path']
    if form.is_submitted():
        # Get the question from the form
        user_question = form.question.data
        # Embedding Model
        embedding_function = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory=doc['vector_db_path'], embedding_function=embedding_function, collection_name=doc['vector_db_name'])
        docs = db.similarity_search(user_question)
        chain = get_langchain_chain()
        response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)
        print(response)
    return render_template('excel/query_excel.html', title="DocuSense AI: Query Excel", form=form, excel_path=excel_path, answer=response)
  

# Delete Excel from Database
@upload_excel_route.route('/delete_excel/<string:docid>', methods=['GET', 'POST'])
def delete_excel(docid):
    # Create Connection to MongoDB database
    db = db_client.get_database('documents')
    # Get the collection
    collection = db.get_collection('excel')
    # Get the document by ID
    doc = collection.find_one({"_id": ObjectId(docid)})
    # Get the filename
    filename = doc['filename']
    # Delete the vector database from the static/vector/pdf folder
    vector_db_path = doc['vector_db_path']
    vector_db_path = os.path.join(VECTOR_DB_PATH, vector_db_path)
    shutil.rmtree(vector_db_path)
    # Delete the document from the collection
    collection.delete_one({"_id": ObjectId(docid)})
    flash(f'Excel File {filename} deleted successfully.', 'success')
    return redirect('/upload_excel')