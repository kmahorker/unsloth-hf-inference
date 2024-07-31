FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the dependencies from requirements.txt
RUN pip install -r requirements.txt \
    && pip install "unsloth[cu121-ampere-torch211] @ git+https://github.com/unslothai/unsloth.git" \
    && pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# Copy the rest of your application code
COPY inference inference

CMD ["uvicorn", "inference.server:app", "--host", "0.0.0.0", "--port", "80"]