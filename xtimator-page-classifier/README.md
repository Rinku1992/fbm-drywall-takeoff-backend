# Drywall Page Classifier

In-house ConvNeXt-Small page classifier. Replaces the **classification responsibility** of the Gemini call in `drywall-takeoff-3d/helper.py:classify_plan`. Gemini continues to produce `mask_factor` and `bounding_box_offsets` via a modified (shorter) prompt. The two services are called **in parallel** by the caller, results are merged before being returned to the UI.

## Contract

### Endpoint
```
POST /classify_pages
Content-Type: application/json
```

Internal-only (Cloud Run ingress = `internal-and-cloud-load-balancing`). Authenticated via signed ID token, same pattern as wall-detector.

### Input — JSON body

| Field | Type | Description |
|---|---|---|
| `project_id` | string | Project ID; lowercased for GCS path lookup |
| `plan_id` | string | Plan ID; lowercased for GCS path lookup |
| `user_id` | string | User ID; logged but not used for path resolution |

> The PDF is fetched from GCS at `gs://drywall-takeoff-artifacts-dev/{project_id_lower}/{plan_id_lower}/floor_plan.PDF` by the service. Caller does not send images. The response contains one entry per page in the PDF (no caller-chosen subset).

### Output — JSON

```json
{
  "pages": [
    {
      "page_number": 0,
      "plan_type": "FLOOR_PLAN",
      "confidence": 0.987,
      "model_prediction": "FLOOR_PLAN",
      "threshold_demoted": false
    },
    {
      "page_number": 1,
      "plan_type": "ELEVATION_PLAN",
      "confidence": 0.823,
      "model_prediction": "ELEVATION_PLAN",
      "threshold_demoted": false
    }
  ],
  "inference_time_seconds": 1.24,
  "total_time_seconds": 1.50
}
```

`plan_type` strings match `drywall-takeoff-3d/prompts.py:planType` exactly — `FLOOR_PLAN`, `ROOF_PLAN`, `ELECTRICAL_PLAN`, `FOUNDATION_PLAN`, `ELEVATION_PLAN`, `NOT_ARCHITECTURAL_PLAN`.

`threshold_demoted = true` means the raw model output was `FLOOR_PLAN` but confidence was below 0.85, so we demoted to the second-best class. Useful for monitoring how aggressive the threshold is.

### Error responses

| Status | Reason |
|---|---|
| 404 | PDF not found in GCS at the resolved path |
| 503 | Service not ready (model loading) |
| 500 | GCS download failed, PDF metadata read failed, page rendering failed, or inference error |

## Architecture decisions baked in

| Decision | Choice | Why |
|---|---|---|
| Model arch | ConvNeXt-Small (50M params) | Highest val F1 in training; fits 16GB L4 easily |
| Image size | 640×640 (resized inside the service) | Matches training; caller doesn't resize |
| Inference | 5-view tile-max | Catches title-block in corner crops |
| Threshold | 0.85 on floor_plan | Test set: 5/5 FPs demoted, 0 TPs killed |
| Checkpoint loading | Baked into image | Lowest cold-start latency |
| Container concurrency | 1 (1 batch per instance) | GPU saturates on 5-view tile-max |
| Auth | Signed ID token | Matches existing pattern with wall-detector |

## File structure

```
drywall-page-classifier/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + /classify_pages endpoint
│   ├── inference.py         # ConvNeXt loader + tile-max + threshold
│   ├── schemas.py           # Pydantic request/response
│   └── config.py            # Class names, threshold, paths
├── checkpoints/
│   └── best.pth             # YOU put your trained model here before deploy
├── Dockerfile               # Multi-stage CUDA 12.4 image
├── requirements.txt
├── service.yaml             # Cloud Run + L4 GPU declaration
├── deploy.sh                # One-shot build + deploy
├── test_local.py            # Smoke tester
└── README.md
```

## Deployment

### 1. Place the trained model

```bash
cp /home/jupyter/floorplan_classifier/checkpoints_v2/best.pth \
   ./checkpoints/best.pth
```

This file is baked into the Docker image at build time.

### 2. Verify L4 GPU quota

```bash
gcloud compute project-info describe --project=prj-fbm-drywall-dev | grep -A 2 "nvidia.*l4"
```

If quota is 0 or insufficient, request increase via GCP Console → IAM & Admin → Quotas.

### 3. Build and deploy

```bash
chmod +x deploy.sh
./deploy.sh
```

This builds via Cloud Build (no local Docker needed), pushes to gcr.io, and applies `service.yaml`. First build takes 10-15 min (PyTorch CUDA wheels). Subsequent builds use layer cache — 3-5 min.

The script prints the service URL at the end. Add it to `drywall-takeoff-3d/config/gcp.yaml`.

### 4. Smoke test

```bash
URL=$(gcloud run services describe drywall-page-classifier \
        --region=us-central1 --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token)

# Pull a few PNGs from somewhere for testing
mkdir -p ./test_pngs
# (copy 3-5 PNG files into ./test_pngs/)

python test_local.py --url $URL --token $TOKEN \
    --image_dir ./test_pngs --pages 0,1,2,3,4
```

### 5. Watch logs

```bash
gcloud run services logs tail drywall-page-classifier --region=us-central1
```

## Performance reference

On 1× L4 GPU (the Cloud Run target):
- 1 page: ~250ms inference (5-view tile-max + threshold)
- 10 pages: ~2.5s inference, ~3-4s total request (including upload)
- Model load + warmup at cold start: ~5-10s

## Cost reference

L4 on Cloud Run: ~$0.67/hour, billed per second.
- `minScale=1` keeps 1 warm 24/7 → ~$16/day baseline
- Set `minScale=0` for true scale-to-zero → ~$0 idle, but +30s cold start on first request after idle

## Rollback

To revert to LLM-only classification in the caller (drywall-takeoff-3d):

```bash
gcloud run services update drywall-takeoff-3d-fbm \
    --region=us-central1 \
    --set-env-vars USE_INHOUSE_CLASSIFIER=false
```

This flips a feature flag in the caller code — see `drywall-takeoff-3d-patch/` for details.

## Monitoring to set up post-deploy

After deploy, in Cloud Console → Monitoring:
- `run.googleapis.com/request_count` — request volume
- `run.googleapis.com/request_latencies` — p50, p95, p99
- `run.googleapis.com/container/gpu/utilization` — should hover 50-80% during inference
- Alert on 5xx error rate > 5% sustained for 5 minutes
