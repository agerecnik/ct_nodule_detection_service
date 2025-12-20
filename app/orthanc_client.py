import requests
from requests.auth import HTTPBasicAuth
import logging

from app.settings import settings

logger = logging.getLogger(__name__)


class OrthancClient:
    def __init__(self, base_url: str = settings.ORTHANC_URL):
        self.base_url = base_url.rstrip("/")
        self.auth = HTTPBasicAuth(settings.ORTHANC_USERNAME, settings.ORTHANC_PASSWORD)

    def get_series_instances(self, series_id: str) -> list[str]:
        logger.info(f"Fetching instances for series {series_id}")
        r = requests.get(f"{self.base_url}/series/{series_id}", auth=self.auth, timeout=60)
        r.raise_for_status()
        return r.json()["Instances"]

    def download_instance_file(self, instance_id: str) -> bytes:
        logger.debug(f"Downloading instance {instance_id}")
        r = requests.get(f"{self.base_url}/instances/{instance_id}/file", auth=self.auth, timeout=60)
        r.raise_for_status()
        return r.content

    def upload_instance(self, dicom_bytes: bytes) -> dict:
        logger.info("Uploading SR to Orthanc")
        r = requests.post(f"{self.base_url}/instances", data=dicom_bytes, auth=self.auth, timeout=60)
        r.raise_for_status()
        return r.json()
