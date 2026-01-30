# Deploy vLLM on Azure ML Managed Online Endpoints (Private Network)

This guide walks through deploying a vLLM-based OpenAI-compatible inference server on Azure Machine Learning Managed Online Endpoints in a private network configuration. In private mode, the workspace ACR cannot pull images from public registries, so containers must be built locally and pushed to the workspace ACR.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Azure ML Workspace (Private VNet)                   │
│                                                                             │
│  ┌───────────────────┐      ┌──────────────────────────────────────────┐    │
│  │   Model Registry  │      │          Managed Online Endpoint         │    │
│  │                   │      │              (vllm-private)              │    │
│  │  ┌─────────────┐  │      │                                          │    │
│  │  │ Qwen3-8B:1  │──┼──────┼───┐    ┌──────────────────────────────┐  │    │
│  │  └─────────────┘  │      │   │    │        Deployment            │  │    │
│  └───────────────────┘      │   │    │         (current)            │  │    │
│                             │   │    │                              │  │    │
│  ┌───────────────────┐      │   └────┤  Model mounted at /models    │  │    │
│  │ Container Registry│      │        │                              │  │    │
│  │      (ACR)        │      │   ┌────┤  vLLM container image        │  │    │
│  │                   │      │   │    │                              │  │    │
│  │  ┌─────────────┐  │      │   │    │  GPU: NC24ads_A100_v4        │  │    │
│  │  │  vllm:1.0   │──┼──────┼───┘    │                              │  │    │
│  │  └─────────────┘  │      │        │  Port 8000 (OpenAI API)      │  │    │
│  └───────────────────┘      │        └──────────────────────────────┘  │    │
│                             │                                          │    │
│                             └──────────────────────────────────────────┘    │
│                                              │                              │
└──────────────────────────────────────────────┼──────────────────────────────┘
                                               │
                                               ▼
                              ┌────────────────────────────────┐
                              │     Private Endpoint / VNet    │
                              │   /v1/chat/completions (POST)  │
                              └────────────────────────────────┘
```

| Component | Description |
|-----------|-------------|
| **Model Registry** | Stores the downloaded HuggingFace model (Qwen3-8B) as an Azure ML asset |
| **Container Registry (ACR)** | Hosts the vLLM Docker image, pushed from your local machine |
| **Managed Online Endpoint** | Network-facing resource that routes requests; no compute cost until deployment |
| **Deployment** | Running instance that combines model + container + GPU compute |

## Repository Structure

```
private/
├── README.md
├── LICENSE
├── endpoint.yaml      # Managed online endpoint configuration
├── deployment.yml     # Deployment configuration (model, container, compute)
└── env/
    └── Dockerfile     # vLLM container definition
```

## 0. Prerequisites

### Required Tools

- **Azure CLI** (2.x or later): [Install Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli)
- **Azure ML CLI extension**: Install with `az extension add -n ml -y`
- **Python 3.10+** with pip: [Install Python](https://www.python.org/downloads/)
- **Docker Engine**: [Install Docker Engine](https://docs.docker.com/engine/install/). After installation, complete the [post-installation steps for Linux](https://docs.docker.com/engine/install/linux-postinstall/) to run Docker without `sudo`.

### Environment Variables

Set these variables for use throughout the deployment:

```bash
RG=<your-resource-group>
WS=<your-workspace-name>
ACR_NAME=<your-acr-name>  # e.g., the ACR attached to your Azure ML workspace
```

To find your workspace ACR name, check the Azure ML workspace properties in the Azure Portal or run:

```bash
az ml workspace show --name $WS --resource-group $RG --query container_registry -o tsv
```

## 1. Model Registration

Download the model from HuggingFace and register it in Azure ML.

### Install HuggingFace CLI

```bash
pip install -U huggingface_hub
```

### Authenticate (if model is gated)

```bash
hf auth login
```

### Download the Model

```bash
mkdir -p ~/models

hf download Qwen/Qwen3-8B \
  --local-dir ~/models/Qwen3-8B \
  --local-dir-use-symlinks False
```

### Register in Azure ML

```bash
az ml model create \
  --name Qwen3-8B \
  --version 1 \
  --path ~/models/Qwen3-8B \
  --type custom_model \
  --resource-group $RG \
  --workspace-name $WS
```

## 2. Build and Push Container Image

In private network mode, the Azure ML workspace ACR cannot pull images from public container registries (Docker Hub, etc.). You must build the image locally and push it to the workspace ACR.

### Dockerfile

The `env/Dockerfile` defines the vLLM container:

```dockerfile
FROM vllm/vllm-openai:latest

ENTRYPOINT python3 -m vllm.entrypoints.openai.api_server --model $MODEL $VLLM_ARGS
```

This runs the vLLM OpenAI-compatible API server. The `$MODEL` and `$VLLM_ARGS` environment variables are set in the deployment configuration.

### Build and Push

```bash
# Login to Azure
az login

# Login to the workspace ACR
az acr login --name $ACR_NAME

# Build the image locally
docker build -t vllm:1.0 env/

# Tag for ACR
docker tag vllm:1.0 $ACR_NAME.azurecr.io/vllm:1.0

# Push to ACR
docker push $ACR_NAME.azurecr.io/vllm:1.0
```

After pushing, note the full image URI. You can use the tag-based reference (`$ACR_NAME.azurecr.io/vllm:1.0`) or get the digest-based reference:

```bash
az acr repository show-manifests --name $ACR_NAME --repository vllm --query "[0].digest" -o tsv
```

## 3. Create the Managed Online Endpoint

The endpoint is the network-facing resource that receives inference requests. Creating an endpoint does not incur compute charges; charges begin when deployments are created.

### endpoint.yaml

```yaml
$schema: https://azuremlsdk2.blob.core.windows.net/latest/managedOnlineEndpoint.schema.json
name: vllm-private
auth_mode: key
public_network_access: disabled
```

| Field | Description |
|-------|-------------|
| `name` | Unique endpoint name within the workspace |
| `auth_mode` | Authentication method (`key` or `aad_token`) |
| `public_network_access` | Set to `disabled` for private network deployments |

### Create the Endpoint

```bash
az ml online-endpoint create \
  --file endpoint.yaml \
  --resource-group $RG \
  --workspace-name $WS
```

## 4. Create the Deployment

The deployment specifies the model, container image, compute instance, and runtime configuration.

### deployment.yml

Update the `image` field with your ACR image URI before deploying.

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: current
endpoint_name: vllm-private
model: "azureml:Qwen3-8B:1"
model_mount_path: /models
environment_variables:
  MODEL: "/models/Qwen3-8B"
  TRANSFORMERS_OFFLINE: "1"
  HF_DATASETS_OFFLINE: "1"
environment:
  image: <your-acr-name>.azurecr.io/vllm:1.0
  inference_config:
    liveness_route:
      port: 8000
      path: /ping
    readiness_route:
      port: 8000
      path: /health
    scoring_route:
      port: 8000
      path: /
instance_type: Standard_NC24ads_A100_v4
instance_count: 1
request_settings:
  request_timeout_ms: 60000
  max_concurrent_requests_per_instance: 16
liveness_probe:
  initial_delay: 10
  period: 10
  timeout: 2
  success_threshold: 1
  failure_threshold: 30
readiness_probe:
  initial_delay: 120
  period: 10
  timeout: 2
  success_threshold: 1
  failure_threshold: 30
egress_public_network_access: disabled
```

| Field | Description |
|-------|-------------|
| `model` | Reference to the registered Azure ML model |
| `model_mount_path` | Path where the model is mounted in the container |
| `environment_variables.MODEL` | Path passed to vLLM to load the model |
| `environment_variables.TRANSFORMERS_OFFLINE` | Prevents runtime downloads (required for private network) |
| `image` | Full ACR image URI (update before deploying) |
| `instance_type` | GPU VM SKU (A100 recommended for large models) |
| `readiness_probe.initial_delay` | Time to wait before first probe (model loading takes time) |
| `egress_public_network_access` | Set to `disabled` to prevent outbound internet access from the deployment |

### Create the Deployment

```bash
az ml online-deployment create \
  --file deployment.yml \
  --resource-group $RG \
  --workspace-name $WS \
  --all-traffic
```

The `--all-traffic` flag routes 100% of traffic to this deployment immediately.

## 5. Test the Endpoint

Once the deployment is ready, test it using the OpenAI-compatible chat completions API.

### Get the Endpoint URL and Key

```bash
# Get scoring URI
az ml online-endpoint show \
  --name vllm-private \
  --resource-group $RG \
  --workspace-name $WS \
  --query scoring_uri -o tsv

# Get API key
az ml online-endpoint get-credentials \
  --name vllm-private \
  --resource-group $RG \
  --workspace-name $WS \
  --query primaryKey -o tsv
```

### Send a Request

```bash
curl -sS "https://<your-endpoint-name>.<region>.inference.ml.azure.com/v1/chat/completions" \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen3-8B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a paragraph summary of vLLM."}
    ],
    "temperature": 0.2,
    "max_tokens": 2048
  }'
```

The `model` field in the request body must match the model path configured in the deployment (`/models/Qwen3-8B`).
