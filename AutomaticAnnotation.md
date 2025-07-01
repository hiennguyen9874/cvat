# Integrate custom YOLOv8 && YOLO11 model into CVAT for automatic annotation

## Start

1. Start CVAT together with the plugin use for AI automatic annotation assistant.

```bash
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```

2. Install nuctl

```bash
wget https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64
```

3. After downloading the nuclio, give it a proper permission and do a softlink

```bash
sudo chmod +x nuctl-1.13.0-linux-amd64
sudo ln -sf $(pwd)/nuctl-1.13.0-linux-amd64 /usr/local/bin/nuctl
```

4. Build the docker image and run the container. After it is done, you can use the model right away in the CVAT.
   ```
   docker pull ultralytics/ultralytics:latest
   DOCKER_BUILDKIT=1 ./serverless/deploy_gpu.sh yolov8
   ```

## Stop

- `docker ps | grep nuclio`
- `docker stop 0570c7cb814a`
- `nuctl delete function yolov8 --namespace nuclio`
