import numpy as np
import pydicom
from pydicom import Dataset
from pydicom.uid import generate_uid
from pydicom.sr.codedict import codes
import highdicom as hd
from highdicom.sr.templates import ObservationContext
from highdicom.sr.templates import ObserverContext
from highdicom.sr.content import ImageRegion
from highdicom.sr.coding import CodedConcept
from highdicom.sr import TrackingIdentifier
import logging

from app.dicom_series import SeriesVolume
from app.inference_monai_bundle import Detection
from app.settings import settings

logger = logging.getLogger(__name__)


def build_tid1500_sr(series: SeriesVolume, detections: list[Detection]) -> pydicom.Dataset:
    logger.info(f"Building TID1500 SR | detections={len(detections)}")
    if not detections:
        raise ValueError("No detections to store")

    observer_device_context = ObserverContext(
        observer_type=codes.DCM.Device,
        observer_identifying_attributes=hd.sr.DeviceObserverIdentifyingAttributes(
            uid=generate_uid(),
            name="CT Nodule AI Service"
        )
    )

    obs_ctx = ObservationContext(
        observer_device_context=observer_device_context
    )

    measurement_groups = []

    for idx, det in enumerate(detections, start=1):
        logger.debug(
            f"Detection {idx} | score={det.score:.3f} | "
            f"slices={det.xyz_min[2]}..{det.xyz_max[2]}"
        )
        for slice_index in range(det.xyz_min[2], det.xyz_max[2] + 1):

            src_ds = series.ds_by_slice[slice_index]
            ref = hd.sr.SourceImageForRegion(
                referenced_sop_class_uid=src_ds.SOPClassUID,
                referenced_sop_instance_uid=src_ds.SOPInstanceUID,
            )

            x_min_new, x_max_new, y_min_new, y_max_new = expand_2d_roi(det, src_ds, settings.SCALE_2D_ROI)
            region = ImageRegion(
                graphic_type=hd.sr.GraphicTypeValues.POLYLINE,
                graphic_data=np.array([
                    [x_min_new, y_min_new],
                    [x_max_new, y_min_new],
                    [x_max_new, y_max_new],
                    [x_min_new, y_max_new],
                    [x_min_new, y_min_new],
                ], dtype=float),
                source_image=ref,
            )

            finding = CodedConcept(value="396006", scheme_designator="SCT", meaning="Pulmonary nodule")

            tracking_id = TrackingIdentifier(
                identifier=f"AI_NODULE_{generate_uid()}",
                uid=generate_uid(),
            )

            mg = hd.sr.PlanarROIMeasurementsAndQualitativeEvaluations(
                referenced_region=region,
                tracking_identifier=tracking_id,
                finding_type=finding,
                measurements=[
                    hd.sr.Measurement(
                        name=CodedConcept("R-404FB", "SRT", "Probability"),
                        value=float(det.score * 100),
                        unit=CodedConcept("%", "UCUM", "percent")
                    )
                ],
                qualitative_evaluations=[],
            )
            measurement_groups.append(mg)

    measurement_report = hd.sr.MeasurementReport(
        observation_context=obs_ctx,
        procedure_reported=codes.LN.CTUnspecifiedBodyRegion,
        imaging_measurements=measurement_groups,
        title=codes.DCM.ImagingMeasurementReport,
    )

    sr_dataset = hd.sr.ComprehensiveSR(
        evidence=series.ds_by_slice,
        content=measurement_report,
        series_number=1,
        series_instance_uid=generate_uid(),
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="CT Nodule Detection",
        series_description="CT Nodule AI SR",
        is_complete=True,
        is_verified=False
    )

    return sr_dataset

def expand_2d_roi(detection: Detection, src_ds: Dataset, scale: float):
    x_min = detection.xyz_min[0]
    x_max = detection.xyz_max[0]
    y_min = detection.xyz_min[1]
    y_max = detection.xyz_max[1]

    logger.debug(
        f"Expanding ROI | scale={scale} | "
        f"x={x_min}..{x_max} y={y_min}..{y_max}"
    )

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    w *= scale
    h *= scale

    x_min_new = x_center - w / 2
    x_max_new = x_center + w / 2
    y_min_new = y_center - h / 2
    y_max_new = y_center + h / 2

    rows = int(src_ds.Rows)
    cols = int(src_ds.Columns)

    x_min_new = max(1, round(x_min_new))
    y_min_new = max(1, round(y_min_new))
    x_max_new = min(cols, round(x_max_new))
    y_max_new = min(rows, round(y_max_new))

    logger.debug(
        f"Scaled ROI x={x_min_new}..{x_max_new} y={y_min_new}..{y_max_new}"
    )

    return x_min_new, x_max_new, y_min_new, y_max_new
