import datasets
from tango import Step

from textual_inversion.transforms import TransformExamples


@Step.register("textual_inversion::transform_data")
class TransformData(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: bool = True

    def run(  # type: ignore
        self,
        dataset: datasets.DatasetDict,
        example_transformer: TransformExamples,
    ) -> datasets.DatasetDict:
        dataset = dataset.with_transform(example_transformer)
        return dataset
