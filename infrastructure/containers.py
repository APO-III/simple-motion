from dependency_injector import containers, providers
from infrastructure.settings import settings
from data_features.use_cases.features_service import FeaturesService


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
