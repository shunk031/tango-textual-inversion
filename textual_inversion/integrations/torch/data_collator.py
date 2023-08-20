from typing import Any, Dict, List

import torch
from tango.integrations.torch import DataCollator


@DataCollator.register("textual_inversion::custom_collator")
class CustomCollator(DataCollator[Dict[str, Any]]):
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "pixel_values": torch.cat(
                [item["pixel_values"].unsqueeze(dim=0) for item in batch], dim=0
            ),
            "input_ids": torch.cat(
                [item["input_ids"].unsqueeze(dim=0) for item in batch], dim=0
            ),
        }
