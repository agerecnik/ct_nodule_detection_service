import io
import tempfile
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pydicom
import SimpleITK as sitk
import logging

logger = logging.getLogger(__name__)


@dataclass
class SeriesVolume:
    itk_image: sitk.Image
    ds_by_slice: list[pydicom.Dataset]


def slice_position_along_normal(ds: pydicom.Dataset) -> float:
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    row = iop[:3]
    col = iop[3:]
    normal = np.cross(row, col)
    ipp = np.array(ds.ImagePositionPatient, dtype=float)
    return float(np.dot(ipp, normal))


def build_series_volume(dicom_files: list[bytes]) -> SeriesVolume:
    logger.info(f"Building series volume from {len(dicom_files)} DICOM files")
    if not dicom_files:
        raise ValueError("No DICOM files provided")

    dss: list[pydicom.Dataset] = []
    bytes_by_uid: dict[str, bytes] = {}

    for b in dicom_files:
        ds = pydicom.dcmread(io.BytesIO(b), force=True)
        sop = str(ds.SOPInstanceUID)
        dss.append(ds)
        bytes_by_uid[sop] = b

    try:
        dss_sorted = sorted(dss, key=slice_position_along_normal)
    except Exception:
        dss_sorted = sorted(dss, key=lambda x: float(x.ImagePositionPatient[2]))

    sop_uids = [str(ds.SOPInstanceUID) for ds in dss_sorted]

    with tempfile.TemporaryDirectory() as tmp:
        dpath = Path(tmp)

        for i, uid in enumerate(sop_uids):
            (dpath / f"{i:06d}.dcm").write_bytes(bytes_by_uid[uid])

        reader = sitk.ImageSeriesReader()
        series_files = [str(dpath / f"{i:06d}.dcm") for i in range(len(sop_uids))]
        reader.SetFileNames(series_files)
        itk_img = reader.Execute()
        logger.info(f"Volume built | size={itk_img.GetSize()} | spacing={itk_img.GetSpacing()}")

    return SeriesVolume(itk_image=itk_img, ds_by_slice=dss_sorted)
