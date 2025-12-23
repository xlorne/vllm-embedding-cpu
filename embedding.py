from transformers import AutoTokenizer, AutoModel
import torch
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union

app = FastAPI()

# 全局变量：记录模型加载时间（可选）
MODEL_LOAD_START = time.perf_counter()
model_name = "/models/bge-m3"
max_length = 8192

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
model.to("cpu")
MODEL_LOAD_END = time.perf_counter()
LOAD_DURATION_NS = int((MODEL_LOAD_END - MODEL_LOAD_START) * 1e9)  # 纳秒

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    encoding_format: str = "float"

@app.post("/api/embed")
async def embeddings(req: EmbeddingRequest):
    start_time = time.perf_counter()  # 请求开始时间

    try:
        if isinstance(req.input, str):
            texts = [req.input]
        else:
            texts = req.input

        if not texts:
            raise HTTPException(status_code=400, detail="Input list is empty")

        # Tokenize
        batch_dict = tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

        # 计算 prompt token 数量（用于模拟 prompt_eval_count）
        prompt_token_count = sum(len(ids) for ids in batch_dict['input_ids'])

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = outputs.last_hidden_state

        # Mean pooling with attention mask
        attention_mask = batch_dict['attention_mask']
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        # L2 normalize
        normalized_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)

        # Convert to list of lists
        embedding_list = normalized_embeddings.cpu().numpy().tolist()

        end_time = time.perf_counter()
        total_duration_ns = int((end_time - start_time) * 1e9)

        return {
            "model": "bge-m3",  # 你可以改成 "nomic-embed-text" 或其他名称
            "embeddings": embedding_list,
            "total_duration": total_duration_ns,
            "load_duration": LOAD_DURATION_NS,  # 模型加载耗时（纳秒）
            "prompt_eval_count": prompt_token_count  # 输入 token 总数
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")