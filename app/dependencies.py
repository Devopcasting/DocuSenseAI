from motor.motor_asyncio import AsyncIOMotorClient
from app.config import get_settings

def get_mongodb_client():
    settings = get_settings()
    return AsyncIOMotorClient(settings.mongodb_uri)
