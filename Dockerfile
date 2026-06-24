FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN pip install --upgrade pip

RUN pip install \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    xformers \
    runpod

WORKDIR /app

COPY handler.py /app/handler.py

CMD ["python", "handler.py"]
