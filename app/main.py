from contextlib import asynccontextmanager
import os
import tempfile
import io
import SimpleITK as sitk
import pydicom
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from app.orthanc_client import OrthancClient
from app.dicom_series import build_series_volume
from app.inference_monai_bundle import MonaiLungNoduleDetector
from app.sr_tid1500 import build_tid1500_sr
from app.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("ct_nodule_service")


class RunRequest(BaseModel):
    series_id: str
    upload_sr_to_orthanc: bool = True


detector: MonaiLungNoduleDetector | None = None
orthanc = OrthancClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    logger.info("Initializing MONAI lung nodule detector")
    detector = MonaiLungNoduleDetector()
    logger.info("Detector initialized successfully")
    yield
    logger.info("Shutting down detector")
    detector = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run_inference(req: RunRequest):
    logger.info(f"Run request received | series_id={req.series_id}")
    global detector
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")

    try:
        instance_ids = orthanc.get_series_instances(req.series_id)
        logger.info(f"Found {len(instance_ids)} instances in series {req.series_id}")
        if not instance_ids:
            raise HTTPException(status_code=404, detail=f"No instances found for series {req.series_id}")

        dicom_bytes_list = [orthanc.download_instance_file(iid) for iid in instance_ids]
        series = build_series_volume(dicom_bytes_list)

        with tempfile.TemporaryDirectory() as tmp:
            nifti_path = os.path.join(tmp, "series.nii.gz")
            logger.debug(f"Writing temporary NIfTI to {nifti_path}")
            sitk.WriteImage(series.itk_image, nifti_path)

            detections = detector.predict(nifti_path)

            if req.upload_sr_to_orthanc and len(detections) > 0:
                sr_ds = build_tid1500_sr(series, detections)
                buff = io.BytesIO()
                pydicom.dcmwrite(buff, sr_ds, write_like_original=False)
                upload_resp = orthanc.upload_instance(buff.getvalue())

                return {
                    "status": "done",
                    "detections": len(detections),
                    "orthanc_upload": upload_resp,
                }

            return {"status": "done", "detections": len(detections)}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
