# CVAT Deploy

A comprehensive deployment setup for CVAT (Computer Vision Annotation Tool) with integrated serverless AI models for automatic annotation.

## Overview

This project provides a complete deployment solution for CVAT with support for various AI models including YOLOv8, YOLO11, and many other popular computer vision models. The setup includes serverless inference capabilities powered by Nuclio functions.

## Project Structure

```
cvat/
├── components/
│   ├── analytics/          # Analytics and monitoring components
│   │   ├── clickhouse/     # ClickHouse database setup
│   │   ├── grafana/        # Grafana dashboards and configuration
│   │   └── vector/         # Vector data processing
│   └── serverless/         # Serverless components configuration
├── serverless/             # AI model implementations
│   ├── onnx/              # ONNX format models
│   │   └── WongKinYiu/yolov7/
│   ├── openvino/          # OpenVINO optimized models
│   │   ├── base/          # Base OpenVINO components
│   │   ├── dextr/         # DEXTR interactive segmentation
│   │   └── omz/           # Open Model Zoo models
│   │       ├── intel/     # Intel optimized models
│   │       └── public/    # Public pre-trained models
│   ├── pytorch/           # PyTorch based models
│   │   ├── detectron2/    # Facebook's Detectron2 models
│   │   ├── sam/           # Segment Anything Model
│   │   ├── siammask/      # SiamMask tracking
│   │   ├── mmpose/        # Human pose estimation
│   │   └── fbrs/          # Feature-based refinement
│   └── tensorflow/        # TensorFlow models
├── settings/              # Django settings
├── share/                 # Shared resources
└── yolov8/               # YOLOv8 model implementation
```

## Supported Models

### Object Detection

- **YOLOv8** - Latest YOLO architecture for object detection
- **YOLO11** - Next generation YOLO model
- **YOLOv7** - High-performance object detection
- **Faster R-CNN** - Region-based CNN for accurate detection
- **RetinaNet** - Single-stage detector with focal loss
- **Mask R-CNN** - Instance segmentation

### Segmentation

- **SAM (Segment Anything Model)** - Universal image segmentation
- **DEXTR** - Deep Extreme Cut for interactive segmentation
- **FBRS** - Feature-based refinement for segmentation
- **Semantic Segmentation ADAS** - Automotive segmentation

### Tracking & Pose

- **SiamMask** - Visual object tracking
- **TransT** - Transformer-based tracking
- **HRNet** - Human pose estimation

### Specialized Models

- **Face Detection** - Intel optimized face detection
- **Person Re-identification** - Retail person tracking
- **Text Detection** - OCR text detection

## Quick Start

### 1. Start CVAT with Serverless Support

Start CVAT together with the serverless plugin for AI automatic annotation:

```bash
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```

### 2. Install Nuclio CLI

Download and install the Nuclio CLI tool:

```bash
# Download nuclio
wget https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64

# Set permissions and create symlink
sudo chmod +x nuctl-1.13.0-linux-amd64
sudo ln -sf $(pwd)/nuctl-1.13.0-linux-amd64 /usr/local/bin/nuctl
```

### 3. Deploy AI Models

Deploy specific models using the deployment scripts:

#### For GPU-enabled systems:

```bash
# Deploy YOLOv8 model
DOCKER_BUILDKIT=1 ./serverless/deploy_gpu.sh yolov8

# Deploy other models
DOCKER_BUILDKIT=1 ./serverless/deploy_gpu.sh <model_name>
```

#### For CPU-only systems:

```bash
DOCKER_BUILDKIT=1 ./serverless/deploy_cpu.sh <model_name>
```

### 4. Access CVAT

Once deployed, CVAT will be available with automatic annotation capabilities powered by the deployed AI models.

## Model Integration

### Adding Custom Models

1. Create a new directory under the appropriate framework folder (`pytorch/`, `tensorflow/`, `onnx/`, or `openvino/`)
2. Implement the model handler following the existing patterns
3. Create Nuclio function configuration files
4. Add deployment scripts if needed

### Model Handler Structure

Each model requires:

- `main.py` - Nuclio function entry point
- `model_handler.py` - Model loading and inference logic
- `function.yaml` - Nuclio function configuration
- `Dockerfile` - Container build instructions (if custom base needed)

## Configuration

### Analytics and Monitoring

The project includes comprehensive monitoring setup:

- **ClickHouse** - Event data storage
- **Grafana** - Visualization dashboards
- **Vector** - Data pipeline processing

### Custom Settings

Modify `settings/production.py` for production deployment configurations.

## Development

### Prerequisites

- Docker and Docker Compose
- GPU drivers (for GPU-accelerated models)
- Sufficient disk space for model downloads

### Local Development

1. Clone the repository
2. Follow the Quick Start instructions
3. Access logs: `docker compose logs -f`

## Troubleshooting

### Common Issues

- **GPU not detected**: Ensure NVIDIA drivers and Docker GPU support are properly installed
- **Model deployment fails**: Check Docker build logs and ensure sufficient resources
- **Performance issues**: Monitor resource usage and consider model optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add appropriate tests
5. Submit a pull request

## License

This project follows the licensing terms of the original CVAT project.

## Support

For issues related to:

- CVAT core functionality: Visit the [official CVAT repository](https://github.com/opencv/cvat)
- Model integration: Check the specific model documentation in the serverless directory
- Deployment issues: Review the Docker and Nuclio documentation
