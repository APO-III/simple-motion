from dependency_injector import containers, providers
from infrastructure.settings import settings
from data_features.use_cases.features_service import FeaturesService
from pose_processing.use_cases.pose_extraction_service import PoseExtractionService
from orchestrator.motion_orchestrator import MotionOrchestrator
from labels_reader.use_cases.label_studio_reader import LabelStudioReader
from orchestrator.utils.csv_exporter import CSVExporter 

class Container(containers.DeclarativeContainer):
    """
    Dependency injection container for the application.
    """

    config = providers.Configuration()

    config.from_dict({
        'debug': settings.debug,
        'log_level': settings.log_level,
        'model_confidence_threshold': settings.model_confidence_threshold,
        'mediapipe_model_complexity': settings.mediapipe_model_complexity,
        'target_fps': settings.target_fps
    })

    features_service = providers.Singleton(FeaturesService)

    pose_extraction_service = providers.Singleton(
        PoseExtractionService,
        mediapipe_style=config.mediapipe_model_complexity
    )

    label_studio_reader = providers.Singleton(LabelStudioReader)

    motion_orchestrator = providers.Singleton(
        MotionOrchestrator,
        pose_extraction_service=pose_extraction_service,
        features_service=features_service,
        label_reader=label_studio_reader
    )
    
    exporter = providers.Factory(
        CSVExporter
    )
