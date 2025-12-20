import os
import sys
from dataclasses import dataclass
import nibabel as nib
import numpy as np
import torch
from huggingface_hub import snapshot_download
from monai.bundle import ConfigParser
import logging

from app.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    score: float
    xyz_min: tuple[int, int, int]
    xyz_max: tuple[int, int, int]


class MonaiLungNoduleDetector:

    def __init__(self):
        logger.info(f"Loading MONAI bundle | repo={settings.HF_REPO_ID} | rev={settings.HF_REVISION}")
        self.bundle_dir = snapshot_download(
            repo_id=settings.HF_REPO_ID,
            revision=settings.HF_REVISION,
        )
        if self.bundle_dir not in sys.path:
            sys.path.insert(0, self.bundle_dir)

        cfg_path = os.path.join(
            self.bundle_dir, "configs", "inference.json"
        )

        self.parser = ConfigParser()
        self.parser.read_config(cfg_path)
        self.parser.parse(True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.network = self.parser.get_parsed_content("network")
        self.detector = self.parser.get_parsed_content("detector")

        _ = self.parser.get_parsed_content("detector_ops")

        self.preprocessing = self.parser.get_parsed_content("preprocessing")
        self.postprocessing = self.parser.get_parsed_content("postprocessing")

        ckpt_path = os.path.join(self.bundle_dir, "models", "model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("model", ckpt)

        self.network.load_state_dict(state, strict=True)
        self.network.to(self.device).eval()

        self.detector.network = self.network
        self.detector.to(self.device).eval()

    def world_mm_to_vox(
            self,
            inv_aff: np.ndarray,
            pts_mm: np.ndarray,
            *,
            bundle_mm_is_negated_xy: bool,
    ) -> np.ndarray:
        pts_mm = np.asarray(pts_mm, dtype=np.float64)

        if bundle_mm_is_negated_xy:
            pts_mm = pts_mm.copy()
            pts_mm[:, 0] *= -1.0
            pts_mm[:, 1] *= -1.0

        ones = np.ones((pts_mm.shape[0], 1), dtype=np.float64)
        pts_h = np.concatenate([pts_mm, ones], axis=1)
        vox_h = (inv_aff @ pts_h.T).T
        return vox_h[:, :3]

    @torch.inference_mode()
    def predict(self, image_path: str) -> list[Detection]:
        logger.info("Starting prediction")

        data = {"image": image_path}
        data = self.preprocessing(data)
        logger.info("Preprocessing completed")

        img = data["image"].to(self.device)
        img = img.unsqueeze(0)

        logger.info("Starting inference")
        pred_list = self.detector(img)
        logger.info("Inference completed")
        if not pred_list:
            return []

        pred = self.postprocessing({**pred_list[0], "image": data["image"]})
        logger.info("Postprocessing completed")
        logger.debug(f"Predictions: {pred}")

        boxes = pred.get("box")
        scores = pred.get("label_scores")
        if boxes is None or scores is None:
            return []

        nii = nib.load(image_path)
        aff = nii.affine
        shape = nii.shape

        dets: list[Detection] = []

        for i, (box, sc) in enumerate(zip(boxes, scores)):
            if sc < float(settings.SCORE_THRESHOLD):
                continue

            x_min, y_min, z_min, dx, dy, dz = box

            inv_aff = np.linalg.inv(aff)

            base = np.array([x_min, y_min, z_min], dtype=np.float64)
            size = np.array([dx, dy, dz], dtype=np.float64)

            offsets01 = np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ],
                dtype=np.float64,
            )

            corners_mm = base[None, :] + offsets01 * size[None, :]

            corners_vox = self.world_mm_to_vox(
                inv_aff,
                corners_mm,
                bundle_mm_is_negated_xy=settings.BUNDLE_MM_IS_NEGATED_XY,
            )

            vmin = np.floor(corners_vox.min(axis=0)).astype(int)
            vmax = np.ceil(corners_vox.max(axis=0)).astype(int)
            vmin = np.maximum(vmin, [0, 0, 0])
            vmax = np.minimum(vmax, np.array(shape) - 1)

            dets.append(
                Detection(
                    score=sc,
                    xyz_min=(vmin[0], vmin[1], vmin[2]),
                    xyz_max=(vmax[0], vmax[1], vmax[2])
                )
            )

            logger.debug(
                f"Detection {i} | score={sc:.3f} | "
                f"x={vmin[0]}..{vmax[0]} y={vmin[1]}..{vmax[1]} z={vmin[2]}..{vmax[2]}"
            )

        return dets
