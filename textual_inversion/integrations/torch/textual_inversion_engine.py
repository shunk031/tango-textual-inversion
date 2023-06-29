from typing import Optional, Union

from tango.common import Lazy
from tango.integrations.torch import TorchTrainingEngine, TrainingEngine
from tango.integrations.torch.model import Model
from tango.integrations.torch.optim import LRScheduler, Optimizer
from tango.integrations.torch.train_config import TrainConfig
from textual_inversion.models import StableDiffusionModel

import torch


@TrainingEngine.register("textual_inversion")
class TextualInversionEngine(TorchTrainingEngine):
    def __init__(
        self,
        train_config: TrainConfig,
        model: Union[Model, Lazy[Model]],
        optimizer: Lazy[Optimizer],
        *,
        lr_scheduler: Optional[Lazy[LRScheduler]] = None,
        amp: bool = False,
        max_grad_norm: Optional[float] = None,
        amp_use_bfloat16: Optional[bool] = None
    ) -> None:
        super().__init__(
            train_config,
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            amp=amp,
            max_grad_norm=max_grad_norm,
            amp_use_bfloat16=amp_use_bfloat16,
        )
        assert isinstance(self.model, StableDiffusionModel), type(self.model)

    def _construct_optimizer(self, optimizer: Lazy[Optimizer]) -> Optimizer:
        # Initialize the optimizer
        optim: Optimizer = optimizer.construct(
            params=self.model.text_encoder.get_input_embeddings().parameters()  # type: ignore
        )
        return optim

    def backward(self, loss: torch.Tensor) -> None:
        super().backward(loss)

        grads = self.model.text_encoder.get_input_embeddings().weight.grad  # type: ignore
        index_grads_to_zero = (
            torch.arange(len(self.model.tokenizer)) != self.model.placeholder_token_id  # type: ignore
        )
        grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)
