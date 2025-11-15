from dotenv import load_dotenv
import os
from pathlib import Path


class Settings:
    """
    Application settings loaded from environment variables.
    """

    def __init__(self):
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(dotenv_path=env_path)

        self.debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
        self.log_level: str = os.getenv('LOG_LEVEL', 'INFO')
        self.model_confidence_threshold: float = float(os.getenv('MODEL_CONFIDENCE_THRESHOLD', '0.5'))
        self.mediapipe_model_complexity: int = int(os.getenv('MEDIAPIPE_MODEL_COMPLEXITY', '2'))
        self.target_fps: float = float(os.getenv('TARGET_FPS', '30.0'))

    def __repr__(self):
        return (
            f"Settings(debug={self.debug}, "
            f"log_level={self.log_level}, "
            f"model_confidence_threshold={self.model_confidence_threshold}, "
            f"mediapipe_model_complexity={self.mediapipe_model_complexity}, "
            f"target_fps={self.target_fps})"
        )


settings = Settings()
