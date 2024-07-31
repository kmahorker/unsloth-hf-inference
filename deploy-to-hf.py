# PREREQUISITES
# Ensure you have billing setup on your huggingface account
# And configure your Huggingface Key to include inference endpoints permissions

import asyncio
import httpx
import uuid
import os
from typing import Dict

async def create_inference_endpoint(endpoint_name: str, repo_name: str, hf_username: str):
    HF_API_KEY = os.getenv("HF_API_KEY")
    url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{hf_username}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "accountId": None,
        "compute": {
            "accelerator": "gpu",
            "instanceSize": "x1",
            "instanceType": "nvidia-t4", # [OPTIONAL] update based on model requirements
            "scaling": {
                "maxReplica": 1,
                "minReplica": 0,
                "scaleToZeroTimeout": 15
            }
        },
        "model": {
            "framework": "custom",
            "image": {
                "custom": {
                    "url": "docker.io/kmahorker/unsloth-hf-inference:latest", # [OPTIONAL] REPLACE WITH YOUR CUSTOM DOCKER IMAGE
                    "health_route": "/health",
                }
            },
            "repository": repo_name,
            "revision": None,
            "task": "text-generation",
        },
        "name": endpoint_name,
        "provider": {
            "region": "us-east-1",
            "vendor": "aws"
        },
        "type": "public"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        
        out = response.json()
        if "error" in out:
            raise Exception(out["error"])
        
    ready_endpoint = await poll_endpoint_status(endpoint_name, hf_username)
    
    return ready_endpoint

async def poll_endpoint_status(endpoint_name: str, hf_username: str, max_attempts: int = 200, delay: int = 10) -> Dict:
    HF_API_KEY = os.getenv("HF_API_KEY")
    url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{hf_username}/{endpoint_name}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
    }

    async with httpx.AsyncClient() as client:
        for _ in range(max_attempts):
            response = await client.get(url, headers=headers)
            endpoint_info = response.json()
            
            if "status" not in endpoint_info:
                print("Status not found in endpoint response. Retrying...", endpoint_info)
                await asyncio.sleep(delay)
                continue
            
            if endpoint_info["status"]["state"] == "running":
                return endpoint_info
            
            if endpoint_info["status"]["state"] in ["failed", "updateFailed"]:
                raise Exception(f"Endpoint creation failed: {endpoint_info['status']['errorMessage']}")
            
            await asyncio.sleep(delay)
    
    raise Exception("Endpoint creation timed out")

async def deploy_model(repo_name: str, endpoint_name: str):
    try:
        endpoint_response = await create_inference_endpoint(endpoint_name, repo_name)
        print(f"Endpoint created: {endpoint_response}")
    except Exception as e:
        raise Exception(f"Deployment failed: {e}")
    
if __name__ == "__main__":
    repo_name = "<username>/<repo_name>" # REPLACE WITH YOUR REPO
    endpoint_name = uuid.uuid4().hex # REPLACE WITH YOUR ENDPOINT NAME. This will be used as the name of the endpoint in Huggingface
    hf_username = "" # REPLACE WITH YOUR HF USERNAME
    
    # os.environ["HF_API_KEY"] = "" # REPLACE WITH YOUR HF API KEY
    
    if os.environ["HF_API_KEY"] == "":
        raise Exception("Please set HF_API_KEY env variable. Ensure all required settings are updated in deploy-to-hf.py")
    
    asyncio.run(deploy_model(repo_name, endpoint_name))