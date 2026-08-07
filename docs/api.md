# API Guide

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/model/info` | Model flags |
| POST | `/prompt/segment` | Visual prompt mask |
| POST | `/vqa/predict` | Single-turn VQA |
| POST | `/vqa/conversation` | Multi-turn (session_id) |
| POST | `/explain` | Overlay artifacts |

## Example

```bash
curl -X POST http://localhost:8000/vqa/predict \
  -F file=@data/images/patient_000_img0.png \
  -F question="Is there evidence of a fracture?" \
  -F question_type=auto \
  -F prompt_type=box \
  -F coordinates=40,40,120,120
```

Model is loaded **once** via `lru_cache` in `api/dependencies.py`.
Stack traces are not returned to clients.
