
# CT Lung Nodule Detection Service

This repository contains a lightweight FastAPI-based AI service for lung nodule detection in CT scans.  
The service retrieves DICOM series from Orthanc, runs MONAI bundle inference, and stores the results back to Orthanc as a DICOM Structured Report (TID 1500) that can be visualized in viewers such as OHIF.
<img width="1382" height="749" alt="image" src="https://github.com/user-attachments/assets/1f8e371c-24da-4e97-97dc-52585303f068" />

---

## Overview

The service performs the following steps:

1. Fetches a CT DICOM series from Orthanc using Orthanc's `series_id`
2. Builds a correctly ordered 3D volume from the individual DICOM slices
3. Runs lung nodule detection using a pretrained MONAI bundle from Hugging Face
4. Filters detections by confidence score
5. Creates a DICOM Comprehensive SR TID 1500 with
6. Uploads the SR back to Orthanc

## Run Server

    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

## API

### Health check
    GET /health
#### Example:
    curl http://127.0.0.1:8000/health
### Run inference
    POST /run
#### Body

    {
	    "series_id": "<ORTHANC_SERIES_ID>"
	}

#### Example

    curl -X POST http://127.0.0.1:8000/run -H "Content-Type: application/json" -d "{\"series_id\":\"a6351dec-72bf9d4f-b54aab2b-bd907d1d-2025d7b8\"}"

## Settings
| Variable | Type | Description |
|--------|------|-------------|
| `ORTHANC_URL` | `string` | Orthanc base URL |
| `ORTHANC_USERNAME` | `string` | Orthanc username |
| `ORTHANC_PASSWORD` | `string` | Orthanc password |
| `HF_REPO_ID` | `string` | Hugging Face MONAI bundle repository |
| `HF_REVISION` | `string` | Bundle revision or branch |
| `BUNDLE_MM_IS_NEGATED_XY` | `bool` | Whether MONAI bundle uses negated X/Y world coordinates |
| `SCORE_THRESHOLD` | `float` | Minimum confidence score for detections |
| `SCALE_2D_ROI` | `float` | Scaling factor for 2D ROIs written into SR |
| `LOG_LEVEL` | `string` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
