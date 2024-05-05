from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from configparser import ConfigParser
from app.v1.routes.dashboard.dashboard_route import router as dashboard_router
from app.v1.routes.pdf.upload_pdf_route import router as upload_pdf_router

# Initialize FastAPI
app = FastAPI()

# Read Application configuration file
config = ConfigParser()
config.read("app/appconfig.ini")
# Get the API version
version = config.get("Application", "Version")

# Static Path
app.mount("/static", StaticFiles(directory=f"app/{version}/static"), name="static")

# Define router
app.include_router(dashboard_router, prefix=f"/{version}")
app.include_router(upload_pdf_router, prefix=f"/{version}")

if __name__ == '__main__':
    uvicorn.run("main:app", port=8000, reload=True)