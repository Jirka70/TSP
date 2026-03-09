from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.augmentation.augmented_samples_dto import AugmentedSamplesDTO
from src.types.interfaces.augmentor import IAugmentor


class AugmentationStage:
    def __init__(self, augmentor: IAugmentor) -> None:
        self._augmentor = augmentor

    def run(self, input_dto: AugmentationInputDTO) -> AugmentedSamplesDTO:
        return self._augmentor.run(input_dto)