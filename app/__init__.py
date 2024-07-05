from flask import Flask
from pymongo import MongoClient
import os

# Flask Application Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = '878436c0a462c4145fa59eec2c43a66a'

# Set the static folder path
app.static_folder = 'static'

# Get the MongoDB connection string from the environment variable
mongo_uri = os.environ.get('MONGO_URI')

# Create a MongoClient object
db_client = MongoClient(mongo_uri)

# Import Blueprint routes
from app.dashboard.routes import dashboard_route
from app.pdf.routes import upload_pdf_route
from app.excel.routes import upload_excel_route

app.register_blueprint(dashboard_route)
app.register_blueprint(upload_pdf_route)
app.register_blueprint(upload_excel_route)