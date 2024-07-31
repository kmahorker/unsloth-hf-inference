# Unsloth x HuggingFace Inference Endpoints as a Custom FastAPI Container Image

## Overview

This repository provides a custom FastAPI server for Unsloth Inference, designed to integrate seamlessly with HuggingFace Inference Endpoints. The custom handler leverages unsloth library functions for efficient model inference of unsloth models.

### What are HuggingFace Inference Endpoints?

HuggingFace Inference Endpoints are dedicated, scalable, and secure infrastructure for hosting and serving machine learning models. They offer an easy-to-use interface for deploying models and managing inference tasks, providing enterprise-grade features such as automatic scaling, robust security, and easy integration with existing workflows.

## Repository Structure

- **inference/**
  - `defaults.py`: Contains default configurations and settings.
  - `loading.py`: Handles model loading and preparation.
  - `server.py`: Main server logic for handling requests and responses.
  - `structs.py`: Data structures and type definitions.

- **deploy-docker.sh**: Script for deploying the Docker container.
- **dockerfile**: Docker configuration for building the custom container.
- **deploy-to-hf.py**: Script to deploy the custom handler to HuggingFace Inference Endpoints.
- **requirements.txt**: List of required Python packages.

## Usage

1. **Create a HuggingFace Account:**
   - Sign up at [HuggingFace](https://huggingface.co) and enable billing for your account.
   - Enable [inference endpoints](https://ui.endpoints.huggingface.co/) for your account
   - Create an API token with access to Inference Endpoints.

2. **Configure Deployment:**
- Edit `deploy-to-hf.py` to include:
   `repo_name`: Location of your repository on HuggingFace.
   `endpoint_name`: Name for your endpoint.
   `hf_username`: Your HuggingFace username.

3. **Set Environment Variable:**
    ```bash
    export HF_API_KEY="your_huggingface_api_token"
    ```

4. **Install local dependendencies**
    ```bash
    pip install -r local-requirements.txt
    ```

5. **Deploy to HuggingFace:**
   ```bash
   python3 deploy-to-hf.py
   ```

## Custom Prompts

1. **Add a New Task Type:**
- Add your new `TaskType` to `structs.py`.

2. **Update Prompt Templates:**
- Define a new `PromptTemplate` in `defaults.py` with your custom prompt mapped to the newly created TaskType.

3. **Deploy Custom Docker Container:**
- Follow the instructions under [Deploy Custom Changes](#deploy-custom-changes) to reflect your customizations.

## Deploy Custom Changes

If you make changes to the repository, you need to redeploy a custom Docker container to reflect these updates on your endpoint.

1. **Edit Docker Deployment:**
- Modify `deploy-docker.sh` to replace `"YOUR_DOCKERHUB_USERNAME"` with your DockerHub username.

2. **Run Deploy Docker Script**
    ```bash
    chmod +x deploy-docker.sh
    ./deploy-docker.sh
    ```

2. **Update Docker Image:**
- In the `create_inference_endpoint` function, update the image with your custom Docker image URL:
    ```python
    "custom": {
        "url": "docker.io/YOUR_DOCKERHUB_USERNAME/unsloth-hf-inference:latest"
    }
    ```

## Running Locally

To run the application locally for testing and development, follow these steps:

1. **Setup Unsloth**
Follow Unsloth Docs to install unsloth: https://docs.unsloth.ai

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Start the Server:**
    ```bash
    uvicorn inference.server:app --host 0.0.0.0 --port 80
    ```

## Using HF Endpoints

To use the deployed endpoint, you can find the URL in the HuggingFace UI or in the response after running `deploy-to-hf.py`. Use this URL to make POST requests.

### cURL Request for `/predict`

```bash
curl -X POST <endpoint_url>/predict \
     -H "Content-Type: application/json" \
     -d '{
           "input": {"text": "Your input text here"},
           "task_type": "YOUR_TASK_TYPE",
           "config": {"max_tokens": 150}
         }'
```

**Expected Response:**
```json
{
    "response": "Model output based on the input text"
}
```

### cURL Request for `/chat`

```bash
curl -X POST <endpoint_url>/chat \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [{"role": "user", "content": "Hello, how are you?"}],
           "config": {"max_tokens": 150}
         }'
```

**Expected Response:**
  ```json
  {
    "last_message": "I'm doing well, thank you! How can I assist you today?",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"},
      {"role": "assistant", "content": "I'm doing well, thank you! How can I assist you today?"}
    ]
  }
  ```

## FAQ

### What is Unsloth & how to add Unsloth models to HuggingFace Hub?
For more information on Unsloth and adding models to the HuggingFace Hub, please refer to the [Unsloth Docs](https://docs.unsloth.ai/).

### Do I have to use this with HF Inference Endpoints?
No, this is a generic FastAPI server. You can deploy it on your own infrastructure, but you will need to manage scaling and deployment yourself.

### How do I update an existing endpoint with new changes?
To update an existing endpoint with new changes, you must rebuild and redeploy the Docker container. Follow the steps outlined in the [Deploy Custom Changes](#deploy-custom-changes) section.

### Is there a way to monitor the usage and performance of my endpoint?
Yes, HuggingFace Inference Endpoints provide monitoring tools to track usage, latency, and other key performance metrics. You can access these metrics from your HuggingFace account dashboard.