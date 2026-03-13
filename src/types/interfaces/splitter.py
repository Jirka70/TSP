from abc import ABC, abstractmethod


class ISplitter(ABC):
    @abstractmethod
    def run(self, input_dto: PreprocessingInputDTO, run_ctx: RunContext) -> StepResult[PreprocessedDataDTO]:
        raise NotImplementedError