# AGENTS.md

## Project Purpose
This repository provides a FastAPI service for extracting chart axes and tick values from images using YOLO models.

- Axis geometry (origin, x_end, y_end): keypoint model (`pose` task)
- Tick labels + OCR parsing: detection model (`detect` task) + EasyOCR

## Main Entry Points
- API app: `main.py`
- Core services: `yolo_service.py`
- Local run script: `run_api.sh`
- Standalone axis inference demo: `infer_pose.py`

## Models Used

### Axis model (important)
- Default weights path: `models/axis/best.pt`
- Env override: `YOLO_AXIS_MODEL_PATH`
- Loaded in: `YoloAxisService` (`yolo_service.py`)
- Model type (verified from checkpoint load): `YOLOv8s-pose`
- Task: `pose`
- Classes: `{0: "chart"}`
- Keypoints shape: `[3, 3]`
- Keypoint order is fixed and must stay:
1. `origin`
2. `x_end`
3. `y_end`

### Tick model
- Default weights path: `models/ticks/best.pt`
- Env override: `YOLO_TICKS_MODEL_PATH`
- Loaded in: `YoloTicksService` (`yolo_service.py`)
- Task: `detect`
- Classes: `{0: "x_tick_label", 1: "y_tick_label"}`

## API Endpoints
- `POST /yolo-axis`
- `POST /yolo-axis-file`
- `POST /yolo-ticks`
- `POST /yolo-ticks-file`
- `GET /health`

Auth: HTTP Basic, credentials from `.env` (`ADMIN_USERNAME`, `ADMIN_PASSWORD`, `USER_USERNAME`, `USER_PASSWORD`).

## Runtime and Dependencies
- Python dependencies: `requirements.txt`
- Key libs: `ultralytics`, `opencv-python`, `easyocr`, `fastapi`, `uvicorn`
- Start API:
```bash
./run_api.sh
```

## Configuration Notes
- Axis model default in code:
`os.path.join(os.path.dirname(__file__), "models", "axis", "best.pt")`
- Tick model default in code:
`os.path.join(os.path.dirname(__file__), "models", "ticks", "best.pt")`
- OCR config via env:
`EASYOCR_LANGS`, `EASYOCR_GPU`, `EASYOCR_ALLOWLIST`

## Data/Training Notes
- Axis dataset config: `dataset_yolo/data.yaml`
- Axis keypoint dataset uses one class (`chart`) with 3 keypoints
- Converter from COCO-like point annotations to YOLO pose labels: `yolo_convertet.py`

## Guidance for Future Changes
- Do not change keypoint order (`origin`, `x_end`, `y_end`) without updating:
  - dataset labels
  - `dataset_yolo/data.yaml` (`kpt_shape`/semantics)
  - post-processing in `YoloAxisService.predict_axis`
- If replacing axis weights, verify:
  - `task == pose`
  - 3 keypoints are present
  - class mapping remains compatible (`chart`)
