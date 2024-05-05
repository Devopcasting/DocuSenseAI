from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorClient
from app.dependencies import get_mongodb_client

router = APIRouter()
templates = Jinja2Templates(directory="app/v1/routes/pdf/templates")

@router.get("/pdf/upload_pdf", response_class=HTMLResponse)
async def upload_pdf(request: Request, client_db: AsyncIOMotorClient = Depends(get_mongodb_client)):
    """Connect to Document database"""
    db = client_db.get_database('documents')
    pdf_doc_collection = db.get_collection('pdf')
    # Check if collection is empty
    if await pdf_doc_collection.count_documents({}) == 0:
        document = []
    else:
        documents = await pdf_doc_collection.find({}).to_list(length=None)
        document = documents
    context = {"request": request,
               "title": "DocuSense AI : Upload PDF",
               "pdf_doc_list": document}    
    return templates.TemplateResponse("pdf/upload_pdf.html", context)