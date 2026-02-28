# 必须使用 slim，完美兼容 wasmtime 等底层 C/Rust 扩展包
FROM python:3.11-slim

WORKDIR /app

# 配置上海时区，并清理缓存以减小镜像体积
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && apt-get remove -y tzdata \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件
COPY requirements.txt .

# 直接通过 requirements.txt 安装所有依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

EXPOSE 8000

# 启动服务
CMD["python", "main.py"]
