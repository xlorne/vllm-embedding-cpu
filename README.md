# vLLM Embedding Server - BGE-M3 文本嵌入服务

基于 FastAPI 和 BGE-M3 模型的文本嵌入服务。提供 RESTful API 接口，支持将文本转换为向量嵌入（embeddings）。

## 功能特性

- 🚀 基于 FastAPI 的高性能异步 API 服务
- 🎯 使用 BGE-M3 模型进行文本嵌入
- 🐳 支持 Docker 容器化部署
- 💻 CPU 模式运行，无需 GPU
- 📦 模型本地化部署，支持离线使用
- 📊 支持单条或多条文本批量处理
- 🎨 支持多种编码格式输出

## 技术栈

- **Python 3.10+**
- **FastAPI** - Web 框架
- **Transformers** - Hugging Face 模型库
- **PyTorch** - 深度学习框架
- **Uvicorn** - ASGI 服务器

## 项目结构

```
vllm-embedding-cpu/
├── embedding.py            # 嵌入服务实现
├── pyproject.toml         # 项目配置
├── Dockerfile             # Docker 镜像构建文件
├── docker-compose.yaml    # Docker Compose 配置
├── download-model.sh      # 模型下载脚本
├── package.sh             # Docker 镜像打包脚本
└── models/                # 模型文件目录
    └── bge-m3/
```

## 快速开始

### 1. 环境要求

- Python 3.10 或更高版本
- pip 包管理器

### 2. 安装依赖

```bash
pip install fastapi uvicorn transformers torch torchvision torchaudio
```

### 3. 下载模型

运行模型下载脚本：

```bash
chmod +x download-model.sh
./download-model.sh
```

脚本会自动从 Hugging Face 下载 BGE-M3 模型到 `./models/bge-m3/` 目录。

**注意**：下载模型需要安装 `huggingface-hub`：

```bash
pip install huggingface-hub
```

或者手动从 [Hugging Face](https://huggingface.co/BAAI/bge-m3) 下载模型到 `./models/bge-m3/` 目录。

### 4. 启动服务

#### 方式一：直接运行

```bash
uvicorn embedding:app --host 0.0.0.0 --port 8000
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

### 文本嵌入接口

**端点**: `POST /api/embed`

**请求体**:

```json
{
    "input": "什么是机器学习？",
    "encoding_format": "float"
}
```

或批量处理：

```json
{
    "input": [
        "机器学习是人工智能的一个分支",
        "今天天气很好",
        "机器学习使用算法从数据中学习模式"
    ],
    "encoding_format": "float"
}
```

**参数说明**:
- `input`: 字符串或字符串数组，待嵌入的文本
- `encoding_format`: 编码格式，默认 `"float"`

**响应示例**:

```json
{
    "model": "bge-m3",
    "embeddings": [
        [0.123, -0.456, 0.789, ...]
    ],
    "total_duration": 123456789,
    "load_duration": 9876543210,
    "prompt_eval_count": 15
}
```

**响应说明**:
- `model`: 使用的模型名称
- `embeddings`: 嵌入向量数组，每个文本对应一个向量
- `total_duration`: 请求处理总耗时（纳秒）
- `load_duration`: 模型加载耗时（纳秒）
- `prompt_eval_count`: 输入 token 总数

### 使用示例

#### cURL

**单条文本**:

```bash
curl -X POST "http://localhost:8000/api/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "什么是机器学习？",
    "encoding_format": "float"
  }'
```

**批量文本**:

```bash
curl -X POST "http://localhost:8000/api/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "机器学习是人工智能的一个分支",
      "今天天气很好",
      "机器学习使用算法从数据中学习模式"
    ],
    "encoding_format": "float"
  }'
```

#### Python

```python
import requests

url = "http://localhost:8000/api/embed"

# 单条文本
payload = {
    "input": "什么是机器学习？",
    "encoding_format": "float"
}

# 或批量文本
payload = {
    "input": [
        "机器学习是人工智能的一个分支",
        "今天天气很好",
        "机器学习使用算法从数据中学习模式"
    ],
    "encoding_format": "float"
}

response = requests.post(url, json=payload)
results = response.json()
print(f"模型: {results['model']}")
print(f"嵌入向量维度: {len(results['embeddings'][0])}")
print(f"处理耗时: {results['total_duration'] / 1e9:.2f} 秒")
```

## 配置说明

### 模型路径

默认模型路径为 `/models/bge-m3`，可在 `embedding.py` 中修改：

```python
model_name = "/models/bge-m3"  # 修改为你的模型路径
```

### 最大序列长度

默认最大序列长度为 `8192`，可在 `embedding.py` 中修改：

```python
max_length = 8192  # 修改为你需要的最大长度
```

### 端口配置

默认端口为 `8000`，可通过以下方式修改：

- **直接运行**: `uvicorn embedding:app --host 0.0.0.0 --port <端口号>`
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
cd vllm-embedding-cpu
pip install fastapi uvicorn transformers torch huggingface-hub
./download-model.sh
uvicorn embedding:app --reload  # 开发模式，支持热重载
```

### 项目依赖

核心依赖包：
- `fastapi` - Web 框架
- `uvicorn` - ASGI 服务器
- `transformers` - Hugging Face 模型库
- `torch` - PyTorch 深度学习框架
- `pydantic` - 数据验证

## 技术细节

### 嵌入处理流程

1. **Tokenization**: 使用 BGE-M3 tokenizer 对输入文本进行分词和编码
2. **模型推理**: 通过 BGE-M3 模型获取隐藏层输出
3. **Mean Pooling**: 使用 attention mask 进行平均池化
4. **L2 归一化**: 对嵌入向量进行 L2 归一化，便于相似度计算

### 性能优化

- 使用 `torch.no_grad()` 禁用梯度计算，减少内存占用
- 支持批量处理，提高处理效率
- CPU 模式运行，无需 GPU 支持

## 注意事项

- 模型首次加载可能需要一些时间（通常几分钟），请耐心等待
- 确保有足够的磁盘空间存储模型文件（约几 GB）
- CPU 模式下推理速度较慢，建议用于开发测试或小规模生产环境
- 生产环境建议使用 GPU 加速以获得更好的性能
- 默认最大序列长度为 8192，超过此长度的文本会被截断

## 许可证

请查看项目根目录的 LICENSE 文件（如有）。

## 贡献

欢迎提交 Issue 和 Pull Request！
