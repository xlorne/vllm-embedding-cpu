# embedding.py
import torch
import numpy as np
import random
import time
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union

# =============== 强制确定性 ===============
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)  # 添加这个
# =========================================

app = FastAPI()

# 模型配置
MODEL_NAME = "/models/bge-m3"
MAX_LENGTH = 8192

# 加载模型（记录加载时间）
print("Loading model and tokenizer...")
load_start = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
model.to("cpu")
# 禁用dropout和其他随机操作（即使已经eval，某些模型可能有内部随机性）
model.config.use_cache = True  # 使用缓存确保一致性
load_end = time.perf_counter()
LOAD_DURATION_NS = int((load_end - load_start) * 1e9)
print(f"Model loaded in {(load_end - load_start):.2f} seconds")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    encoding_format: str = "float"  # 保留字段，当前仅支持 float


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest):
    start_time = time.perf_counter()

    # 标准化输入
    if isinstance(req.input, str):
        texts = [req.input]
    else:
        texts = req.input

    if not texts:
        raise HTTPException(status_code=400, detail="Input list is empty")

    try:
        # 方法1：逐个处理每个文本，确保完全一致的结果
        all_embeddings = []
        prompt_token_count = 0
        
        # 逐个处理每个文本，避免批处理中的并行差异
        for text in texts:
            # Tokenize
            batch_dict = tokenizer(
                [text],  # 单个文本，确保一致性
                max_length=MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

            # Count input tokens
            prompt_token_count += len(batch_dict['input_ids'][0])

            # Generate embeddings
            with torch.no_grad():
                # 禁用自动混合精度等可能引入差异的功能
                with torch.autocast(enabled=False, device_type='cpu' if model.device.type == 'cpu' else 'cuda'):
                    outputs = model(**batch_dict)
                    last_hidden = outputs.last_hidden_state  # [1, seq_len, dim]

            # Mean pooling with attention mask
            attention_mask = batch_dict['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            # L2 normalization (required for BGE models)
            normalized_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)

            # Convert to list
            embedding = normalized_embeddings.cpu().numpy().tolist()[0]
            all_embeddings.append(embedding)

        end_time = time.perf_counter()
        total_duration_ns = int((end_time - start_time) * 1e9)

        return {
            "model": "bge-m3",
            "embeddings": all_embeddings,
            "total_duration": total_duration_ns,
            "load_duration": LOAD_DURATION_NS,
            "prompt_eval_count": prompt_token_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

