from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/v1/routes/dashboard/templates")

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    context = {"request": request,
               "title": "DocuSense AI"}
    return templates.TemplateResponse("dashboard/dashboard.html", context)