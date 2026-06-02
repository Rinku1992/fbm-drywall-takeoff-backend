#!/usr/bin/env bash
# deploy.sh — build and deploy drywall-page-classifier to Cloud Run
#
# Prereqs:
#   1. Place trained checkpoint at ./checkpoints/best.pth
#   2. gcloud authenticated with deploy rights
#   3. Cloud Build API enabled
#   4. NVIDIA L4 quota in us-central1
#
# Usage:
#   ./deploy.sh                # uses :latest tag
#   IMAGE_TAG=v1.0 ./deploy.sh  # uses :v1.0 tag

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-prj-fbm-drywall-dev}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="drywall-page-classifier"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:${IMAGE_TAG}"

echo "═══════════════════════════════════════════════════════════"
echo "Deploying ${SERVICE_NAME}"
echo "═══════════════════════════════════════════════════════════"
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Image:    ${IMAGE_URI}"
echo

# ─── Preflight ────────────────────────────────────────────────
if [ ! -f "./checkpoints/best.pth" ]; then
  echo "ERROR: ./checkpoints/best.pth not found."
  echo "       Copy your trained model from floorplan_classifier/checkpoints_v2/:"
  echo "       cp /home/jupyter/floorplan_classifier/checkpoints_v2/best.pth ./checkpoints/"
  exit 1
fi

CKPT_SIZE_MB=$(du -m ./checkpoints/best.pth | cut -f1)
echo "  Checkpoint: ./checkpoints/best.pth (${CKPT_SIZE_MB} MB)"
echo

gcloud config set project "${PROJECT_ID}"

# ─── Build image with Cloud Build ─────────────────────────────
echo "─── Building Docker image (Cloud Build) ───"
gcloud builds submit . \
    --tag "${IMAGE_URI}" \
    --timeout=30m \
    --machine-type=e2-highcpu-8

# ─── Deploy ───────────────────────────────────────────────────
echo
echo "─── Deploying Cloud Run service ───"

# Substitute the image URI in service.yaml at deploy time
sed "s|gcr.io/prj-fbm-drywall-dev/drywall-page-classifier:latest|${IMAGE_URI}|g" \
    service.yaml > /tmp/service-deploy.yaml

gcloud run services replace /tmp/service-deploy.yaml \
    --project="${PROJECT_ID}" \
    --region="${REGION}"

rm /tmp/service-deploy.yaml

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --format="value(status.url)")

echo
echo "═══════════════════════════════════════════════════════════"
echo "✓ DEPLOY COMPLETE"
echo "═══════════════════════════════════════════════════════════"
echo "  Service URL:  ${SERVICE_URL}"
echo
echo "  Add this URL to drywall-takeoff-3d/config/gcp.yaml:"
echo "    CloudRun:"
echo "        APIs:"
echo "            page_classifier: ${SERVICE_URL}"
echo
echo "  View logs:"
echo "    gcloud run services logs tail ${SERVICE_NAME} --region=${REGION}"
