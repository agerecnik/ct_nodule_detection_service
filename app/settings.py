from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    # Orthanc
    ORTHANC_URL: str = "http://127.0.0.1:8042"
    ORTHANC_USERNAME: str = "orthanc"
    ORTHANC_PASSWORD: str = "orthanc"

    # MONAI model
    HF_REPO_ID: str = "MONAI/lung_nodule_ct_detection"
    HF_REVISION: str = "main"
    BUNDLE_MM_IS_NEGATED_XY: bool = True

    # Inference threshold
    SCORE_THRESHOLD: float = 0.6

    # Resize 2D ROIs
    SCALE_2D_ROI: float = 1.25

    # Logging
    LOG_LEVEL: str = "DEBUG"

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

settings = Settings()
