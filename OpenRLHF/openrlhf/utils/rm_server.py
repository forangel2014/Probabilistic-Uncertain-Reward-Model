import torch
from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


app = FastAPI()
cos_sim_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')


class StringPair(BaseModel):
    str1: str
    str2: str
    
    
def cosine_similarity(str1, str2):
    """
    根据字符串频率计算余弦相似度(0-1)。
    """
    embedding1 = cos_sim_model.encode(str1)
    embedding2 = cos_sim_model.encode(str2)
    similarity = 1 - cosine(embedding1, embedding2)
    
    del embedding1, embedding2
    similarity = float(similarity)
    torch.cuda.empty_cache()
    return similarity

@app.post("/similarity")
async def calculate_similarity(strings: StringPair):
    similarity = cosine_similarity(strings.str1, strings.str2)
    return {"similarity": similarity}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)