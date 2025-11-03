import numpy as np
from data_features.utils.vector_operations import subtract_vectors, normalize_vector, calculate_angle
from data_features.domain.motion_features import MotionFeatures


class FeaturesService:
    """
    Service for calculating motion analysis features.
    """

    def calculate_normalized_leg_length(
        self,
        left_hip: np.ndarray,
        left_knee: np.ndarray,
        left_ankle: np.ndarray,
        right_hip: np.ndarray,
        right_knee: np.ndarray,
        right_ankle: np.ndarray
    ) -> float:
        """
        Calculate average hip-to-ankle distance normalized by leg length.
        """
        left_hip_to_knee = np.linalg.norm(subtract_vectors(left_hip, left_knee))
        left_knee_to_ankle = np.linalg.norm(subtract_vectors(left_knee, left_ankle))
        left_leg_length = left_hip_to_knee + left_knee_to_ankle

        right_hip_to_knee = np.linalg.norm(subtract_vectors(right_hip, right_knee))
        right_knee_to_ankle = np.linalg.norm(subtract_vectors(right_knee, right_ankle))
        right_leg_length = right_hip_to_knee + right_knee_to_ankle

        average_leg_length = (left_leg_length + right_leg_length) / 2.0

        left_hip_to_ankle = np.linalg.norm(subtract_vectors(left_hip, left_ankle))
        right_hip_to_ankle = np.linalg.norm(subtract_vectors(right_hip, right_ankle))
        average_hip_to_ankle = (left_hip_to_ankle + right_hip_to_ankle) / 2.0

        if average_leg_length == 0:
            return 0.0

        normalized_distance = average_hip_to_ankle / average_leg_length

        return normalized_distance

    def calculate_normalized_shoulder_distance(
        self,
        left_shoulder: np.ndarray,
        right_shoulder: np.ndarray,
        left_hip: np.ndarray,
        right_hip: np.ndarray
    ) -> float:
        """
        Calculate normalized shoulder-to-shoulder distance.
        """
        shoulder_to_shoulder = np.linalg.norm(subtract_vectors(left_shoulder, right_shoulder))

        left_shoulder_to_hip = np.linalg.norm(subtract_vectors(left_shoulder, left_hip))
        right_shoulder_to_hip = np.linalg.norm(subtract_vectors(right_shoulder, right_hip))
        average_torso_length = (left_shoulder_to_hip + right_shoulder_to_hip) / 2.0

        if average_torso_length == 0:
            return 0.0

        normalized_distance = shoulder_to_shoulder / average_torso_length

        return abs(normalized_distance)

    def calculate_normalized_shoulder_vector_xz(
        self,
        left_shoulder: np.ndarray,
        right_shoulder: np.ndarray
    ) -> np.ndarray:
        """
        Calculate normalized shoulder vector projected on XZ plane.
        """
        shoulder_vector = subtract_vectors(right_shoulder, left_shoulder)
        shoulder_vector_xz = np.array([shoulder_vector[0], 0, shoulder_vector[2]])
        normalized_vector = normalize_vector(shoulder_vector_xz)

        return normalized_vector

    def calculate_normalized_ankle_vector_xz(
        self,
        left_ankle: np.ndarray,
        right_ankle: np.ndarray
    ) -> np.ndarray:
        """
        Calculate normalized ankle vector projected on XZ plane.
        """
        ankle_vector = subtract_vectors(right_ankle, left_ankle)
        ankle_vector_xz = np.array([ankle_vector[0], 0, ankle_vector[2]])
        normalized_vector = normalize_vector(ankle_vector_xz)

        return normalized_vector

    def calculate_average_hip_angle(
        self,
        left_shoulder: np.ndarray,
        left_hip: np.ndarray,
        left_knee: np.ndarray,
        right_shoulder: np.ndarray,
        right_hip: np.ndarray,
        right_knee: np.ndarray
    ) -> float:
        """
        Calculate average hip angle from both sides.
        """
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        average_angle = (left_hip_angle + right_hip_angle) / 2.0

        return average_angle

    def calculate_average_knee_angle(
        self,
        left_hip: np.ndarray,
        left_knee: np.ndarray,
        left_ankle: np.ndarray,
        right_hip: np.ndarray,
        right_knee: np.ndarray,
        right_ankle: np.ndarray
    ) -> float:
        """
        Calculate average knee angle from both sides.
        """
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        average_angle = (left_knee_angle + right_knee_angle) / 2.0

        return average_angle

    def extract_all_features(
        self,
        left_shoulder: np.ndarray,
        right_shoulder: np.ndarray,
        left_hip: np.ndarray,
        right_hip: np.ndarray,
        left_knee: np.ndarray,
        right_knee: np.ndarray,
        left_ankle: np.ndarray,
        right_ankle: np.ndarray
    ) -> MotionFeatures:
        """
        Extract all motion features from body landmarks.
        """
        normalized_leg_length = self.calculate_normalized_leg_length(
            left_hip, left_knee, left_ankle,
            right_hip, right_knee, right_ankle
        )



        shoulder_vector = self.calculate_normalized_shoulder_vector_xz(
            left_shoulder, right_shoulder
        )

        ankle_vector = self.calculate_normalized_ankle_vector_xz(
            left_ankle, right_ankle
        )

        average_hip_angle = self.calculate_average_hip_angle(
            left_shoulder, left_hip, left_knee,
            right_shoulder, right_hip, right_knee
        )

        average_knee_angle = self.calculate_average_knee_angle(
            left_hip, left_knee, left_ankle,
            right_hip, right_knee, right_ankle
        )

        return MotionFeatures(
            normalized_leg_length=normalized_leg_length,
            shoulder_vector_x=shoulder_vector[0],
            shoulder_vector_z=shoulder_vector[2],
            ankle_vector_x=ankle_vector[0],
            ankle_vector_z=ankle_vector[2],
            average_hip_angle=average_hip_angle,
            average_knee_angle=average_knee_angle
        )
