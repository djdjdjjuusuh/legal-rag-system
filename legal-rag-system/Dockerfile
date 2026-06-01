# 法律RAG多智能体协作系统
# 基于 DeepSeek API

FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 应用代码
COPY . .

# 数据目录
RUN mkdir -p /app/data/chroma /app/data/chroma_kb /app/data/short_term

EXPOSE 8000

# 启动脚本：创建缓存目录并建立软链接（运行时 volume 挂载后才生效）
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
