# vLLM Server - BGE Reranker Service

基于 FastAPI 和 BGE Reranker v2-m3 模型的文档重排序服务。提供 RESTful API 接口，支持对文档列表进行相关性重排序。

## 功能特性

- 🚀 基于 FastAPI 的高性能异步 API 服务
- 🎯 使用 BGE Reranker v2-m3 模型进行文档重排序
- 🐳 支持 Docker 容器化部署
- 💻 CPU 模式运行，无需 GPU
- 📦 模型本地化部署，支持离线使用

## 技术栈

- **Python 3.10+**
- **FastAPI** - Web 框架
- **Transformers** - Hugging Face 模型库
- **PyTorch** - 深度学习框架
- **Uvicorn** - ASGI 服务器

## 项目结构

```
vllm-server/
├── reranker.py            # 重排序服务实现
├── pyproject.toml         # 项目配置
├── Dockerfile             # Docker 镜像构建文件
├── docker-compose.yaml    # Docker Compose 配置
├── download-model.sh      # 模型下载脚本
├── package.sh             # Docker 镜像打包脚本
└── models/                # 模型文件目录
    └── bge-reranker-v2-m3/
```

## 快速开始

### 1. 环境要求

- Python 3.12 或更高版本
- pip 包管理器

### 2. 安装依赖

```bash
pip install vllm
```

### 3. 下载模型

运行模型下载脚本：

```bash
chmod +x download-model.sh
./download-model.sh
```

或者手动下载模型到 `./models/bge-reranker-v2-m3/` 目录。

### 4. 启动服务

#### 方式一：直接运行

```bash
uvicorn reranker:app --host 0.0.0.0 --port 8000
```

#### 方式二：使用 Docker Compose

```bash
docker-compose up -d
```

#### 方式三：使用 Docker

```bash
# 构建镜像
sh package.sh
# 运行容器
docker-compose up -d 
```

## API 文档

服务启动后，访问以下地址查看交互式 API 文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 重排序接口

**端点**: `POST /v1/rerank`

**请求体**:

```json
{
    "model": "bge-reranker-v2-m3",
    "query": "What is the capital of France?",
    "documents": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
        "Horses and cows are both animals"
    ]
}
```

**响应示例**:

```json
{
    "results": [
        {
            "index": 1,
            "score": 10.285713195800781,
            "text": "The capital of France is Paris."
        },
        {
            "index": 0,
            "score": -6.816523551940918,
            "text": "The capital of Brazil is Brasilia."
        },
        {
            "index": 2,
            "score": -11.034854888916016,
            "text": "Horses and cows are both animals"
        }
    ]
}
```

**响应说明**:
- `results`: 重排序后的文档列表，按相关性分数降序排列
- `index`: 原始文档在输入列表中的索引
- `score`: 相关性分数（越高表示越相关）
- `text`: 文档文本内容

### 使用示例

#### cURL

```bash
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习？",
    "documents": [
      "机器学习是人工智能的一个分支",
      "今天天气很好",
      "机器学习使用算法从数据中学习模式"
    ]
  }'
```

#### Python

```python
import requests

url = "http://localhost:8000/v1/rerank"
payload = {
    "query": "什么是机器学习？",
    "documents": [
        "机器学习是人工智能的一个分支",
        "今天天气很好",
        "机器学习使用算法从数据中学习模式"
    ]
}

response = requests.post(url, json=payload)
results = response.json()
print(results)
```

## 配置说明

### 模型路径

默认模型路径为 `/models/bge-reranker-v2-m3`，可在 `reranker.py` 中修改：

```python
model_name = "/models/bge-reranker-v2-m3"  # 修改为你的模型路径
```

### 端口配置

默认端口为 `8000`，可通过以下方式修改：

- **直接运行**: `uvicorn reranker:app --host 0.0.0.0 --port <端口号>`
- **Docker Compose**: 修改 `docker-compose.yaml` 中的端口映射
- **Docker**: 修改 `-p` 参数

## 开发

### 本地开发

1. 克隆项目
2. 安装依赖
3. 下载模型
4. 运行服务

```bash
git clone <repository-url>
cd vllm-server
pip install -r requirements.txt  # 如果有 requirements.txt
./download-model.sh
uvicorn reranker:app --reload  # 开发模式，支持热重载
```

## 注意事项

- 模型首次加载可能需要一些时间，请耐心等待
- 确保有足够的磁盘空间存储模型文件（约几 GB）
- CPU 模式下推理速度较慢，建议用于开发测试
- 生产环境建议使用 GPU 加速

## 许可证

请查看项目根目录的 LICENSE 文件（如有）。

## 贡献

欢迎提交 Issue 和 Pull Request！

