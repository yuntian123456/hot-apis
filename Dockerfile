# 使用 alpine 极简镜像，基础体积仅十兆级别
FROM python:3.11-alpine

WORKDIR /app

# 安装时区数据、设置上海时区，然后立即卸载 tzdata 包以节省空间
RUN apk add --no-cache tzdata \
    && cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && apk del tzdata

# 先复制依赖列表，利用层缓存机制
COPY requirements.txt .

# 使用 --no-cache-dir 避免产生无用的 pip 缓存文件
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目全部代码
COPY . .

EXPOSE 8000

# 启动命令
CMD ["python", "main.py"]