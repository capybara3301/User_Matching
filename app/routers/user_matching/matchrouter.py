from fastapi import APIRouter
from typing import Dict, Any

from utils.user_matching.utils import UserMatcher
from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load("./model/archive/word2vec-model.pkl")

router = APIRouter()

@router.post("/users")
async def get_matches(user: Dict[str, Any]):
    usermatcher = UserMatcher(model, user)
    
    users = await usermatcher.get_users()

    return users