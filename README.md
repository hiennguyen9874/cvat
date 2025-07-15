# CVAT Deploy with YOLOv8 & YOLO11 Integration

A comprehensive deployment setup for **CVAT (Computer Vision Annotation Tool)** with integrated custom YOLOv8 and YOLO11 models for automatic annotation. This project provides an easy-to-use Docker Compose configuration that enables AI-powered annotation workflows.

## Project Description

**CVAT** is an open-source, web-based tool for annotating images and videos for computer vision tasks. This project extends CVAT with serverless AI model integration, allowing you to:

- Deploy CVAT quickly using Docker Compose
- Integrate custom YOLOv8 and YOLO11 models for automatic object detection
- Leverage serverless inference capabilities powered by Nuclio functions
- Scale AI-assisted annotation workflows efficiently

## Features

- ✅ **Easy CVAT Deployment**: One-command deployment using Docker Compose
- ✅ **YOLOv8 & YOLO11 Integration**: Custom model integration for automatic annotation
- ✅ **Serverless Architecture**: Nuclio-powered serverless inference functions
- ✅ **GPU Support**: Optimized for both CPU and GPU environments
- ✅ **Multiple AI Models**: Support for 20+ pre-configured computer vision models
- ✅ **Analytics & Monitoring**: Built-in Grafana dashboards and ClickHouse analytics
- ✅ **Production Ready**: Scalable configuration for production deployments

## Prerequisites

Before getting started, ensure you have the following installed:

### Required Software

- **Docker Engine**: Version 20.10+

  ```bash
  # Install Docker (Ubuntu/Debian)
  sudo apt update
  sudo apt install docker.io docker-compose-plugin
  ```

- **Docker Compose**: Version 2.0+
  ```bash
  # Verify installation
  docker compose version
  ```

### For GPU Support (Optional but Recommended)

- **NVIDIA GPU Drivers**: Latest stable version
- **NVIDIA Docker Toolkit**: [docs.nvidia.com/datacenter/cloud-native/container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### System Requirements

- **RAM**: Minimum 8GB (16GB+ recommended for GPU inference)
- **Storage**: 20GB+ free space for models and data
- **Network**: Internet connection for model downloads

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd cvat
```

### 2. Start CVAT with Serverless Support

Deploy CVAT together with the serverless plugin for AI automatic annotation:

```bash
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```

This command will:

- Start the CVAT web interface
- Launch the Nuclio serverless platform
- Set up analytics components (ClickHouse, Grafana)
- Prepare the environment for model deployment

### 3. Install Nuclio CLI

Download and configure the Nuclio CLI tool for model deployment:

```bash
# Download Nuclio CLI
wget https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64

# Set permissions and create symlink
sudo chmod +x nuctl-1.13.0-linux-amd64
sudo ln -sf $(pwd)/nuctl-1.13.0-linux-amd64 /usr/local/bin/nuctl

# Verify installation
nuctl version
```

## Integrating YOLOv8 and YOLO11

### Model Configuration

The YOLOv8 and YOLO11 models are pre-configured in the following directories:

```
serverless/
├── yolov8/
│   ├── main.py              # Nuclio function entry point
│   └── function-gpu.yaml    # GPU configuration
└── yolov8-traffic/          # Traffic-specific YOLOv8 variant
    ├── main.py
    └── function-gpu.yaml
```

### Custom Model Weights

To use your own trained weights:

1. **Place model files** in the respective model directories:

   ```bash
   # For YOLOv8
   cp your-yolov8-model.pt serverless/yolov8/

   # For traffic-specific model
   cp your-traffic-model.pt serverless/yolov8-traffic/
   ```

2. **Update model paths** in `main.py` files to reference your custom weights.

### Configuration Files

Key configuration files that may need modification:

- **`function.yaml`** / **`function-gpu.yaml`**: Nuclio function specifications
- **`main.py`**: Model loading and inference logic
- **`docker-compose.yml`**: Core CVAT services
- **`components/serverless/docker-compose.serverless.yml`**: Serverless components

### Deploy Models

#### For GPU-enabled Systems:

```bash
# Pull required base image
docker pull ultralytics/ultralytics:latest

# Deploy YOLOv8 model
DOCKER_BUILDKIT=1 ./serverless/deploy_gpu.sh yolov8

# Deploy traffic-specific YOLOv8 model
DOCKER_BUILDKIT=1 ./serverless/deploy_gpu.sh yolov8-traffic
```

#### For CPU-only Systems:

```bash
# Deploy without GPU acceleration
DOCKER_BUILDKIT=1 ./serverless/deploy_cpu.sh yolov8
DOCKER_BUILDKIT=1 ./serverless/deploy_cpu.sh yolov8-traffic
```

## Usage Guide

### Accessing CVAT

1. **Web Interface**: Navigate to [http://localhost:8080](http://localhost:8080)
2. **Default Credentials**:
   - Username: `admin`
   - Password: Check CVAT documentation for default setup

### Using Auto-Annotation

1. **Create a New Project** in CVAT
2. **Upload Images/Videos** to your project
3. **Navigate to the Task** and click on "Actions" → "Automatic Annotation"
4. **Select Your Model**:
   - Choose `yolov8` for general object detection
   - Choose `yolov8-traffic` for traffic-specific detection
5. **Configure Parameters**:
   - Set confidence threshold (default: 0.5)
   - Choose specific classes if needed
6. **Run Annotation** and review results

### Monitoring and Analytics

- **Nuclio Dashboard**: [http://localhost:8070](http://localhost:8070)
- **Grafana Analytics**: [http://localhost:3000](http://localhost:3000)
- **CVAT Admin Panel**: [http://localhost:8080/admin](http://localhost:8080/admin)

## Example Commands

### Docker Compose Operations

```bash
# Start all services
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d

# View logs
docker compose logs -f

# Stop all services
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml down

# Rebuild and restart
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d --build
```

### Model Management

```bash
# List deployed functions
nuctl get functions --namespace nuclio

# Check function logs
nuctl get logs <function-name> --namespace nuclio

# Delete a function
nuctl delete function yolov8-traffic --namespace nuclio

# Redeploy a model
DOCKER_BUILDKIT=1 ./serverless/deploy_gpu.sh yolov8
```

### Testing Models

```bash
# Test YOLOv8 function directly
curl -X POST http://localhost:8080/api/v1/functions/yolov8/call \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-encoded-image>"}'
```

### System Monitoring

```bash
# Check running containers
docker ps | grep -E "(cvat|nuclio)"

# Monitor resource usage
docker stats

# Check GPU usage (if available)
nvidia-smi
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Models Not Appearing in CVAT

**Problem**: Deployed models don't show up in the automatic annotation menu.

**Solutions**:

```bash
# Check if functions are running
nuctl get functions --namespace nuclio

# Restart serverless components
docker compose restart cvat_server nuclio
```

#### 2. GPU Not Detected

**Problem**: Models fall back to CPU despite GPU availability.

**Solutions**:

```bash
# Verify GPU access in containers
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check Docker GPU support
docker info | grep -i gpu
```

#### 3. Out of Memory Errors

**Problem**: Models fail to load due to insufficient memory.

**Solutions**:

- Reduce batch size in model configuration
- Use CPU deployment for large models
- Increase Docker memory limits

#### 4. Slow Inference Performance

**Problem**: Annotation takes too long to complete.

**Solutions**:

- Ensure GPU acceleration is working
- Reduce image resolution for inference
- Optimize model configuration parameters

#### 5. Function Deployment Fails

**Problem**: Model deployment scripts fail with build errors.

**Solutions**:

```bash
# Check Docker build logs
DOCKER_BUILDKIT=1 ./serverless/deploy_gpu.sh yolov8 2>&1 | tee build.log

# Clean Docker cache
docker system prune -a

# Verify base image availability
docker pull ultralytics/ultralytics:latest
```

### Getting Help

- **Check Logs**: Always start by examining container logs
- **Nuclio Dashboard**: Use [localhost:8070](http://localhost:8070) for function debugging
- **Resource Monitoring**: Monitor CPU, memory, and GPU usage
- **Network Issues**: Verify all ports are accessible and not blocked by firewall

## Supported Models

Beyond YOLOv8 and YOLO11, this setup supports 20+ additional models:

### Object Detection

- Faster R-CNN, RetinaNet, Mask R-CNN
- YOLOv7, SSD MobileNet

### Segmentation

- SAM (Segment Anything Model)
- DEXTR Interactive Segmentation
- Semantic Segmentation ADAS

### Tracking & Pose

- SiamMask, TransT
- HRNet Human Pose Estimation

### Specialized

- Face Detection, Text Detection
- Person Re-identification

## License

This project follows the licensing terms of the original [CVAT project](https://github.com/opencv/cvat). Please refer to the CVAT repository for detailed license information.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For support and questions:

- **CVAT Issues**: [Official CVAT Repository](https://github.com/opencv/cvat)
- **Model Integration**: Check the `serverless/` directory documentation
- **Deployment Issues**: Review Docker and Nuclio documentation

---

**Ready to get started?** Follow the setup instructions above and have your AI-powered annotation workflow running in minutes! 🚀
