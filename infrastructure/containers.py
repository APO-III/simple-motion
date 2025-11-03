from dependency_injector import containers, providers
from infrastructure.settings import settings
from data_features.use_cases.features_service import FeaturesService
from pose_processing.use_cases.pose_extraction_service import PoseExtractionService
from orchestrator.motion_orchestrator import MotionOrchestrator


class Container(containers.DeclarativeContainer):
    """
    Dependency injection container for the application.
    """

    config = providers.Configuration()

    config.from_dict({
        'debug': settings.debug,
        'log_level': settings.log_level,
        'model_confidence_threshold': settings.model_confidence_threshold,
        'mediapipe_model_complexity': settings.mediapipe_model_complexity
    })

    features_service = providers.Singleton(FeaturesService)

    pose_extraction_service = providers.Singleton(
        PoseExtractionService,
        mediapipe_style=config.mediapipe_model_complexity
    )

    motion_orchestrator = providers.Singleton(
        MotionOrchestrator,
        pose_extraction_service=pose_extraction_service,
        features_service=features_service
    )
