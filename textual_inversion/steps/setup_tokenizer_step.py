from typing import Optional

from tango import Step
from tango.integrations.transformers import Tokenizer


@Step.register("textual_inversion::setup_tokenizer")
class SetupTokenizer(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True

    def run(  # type: ignore
        self,
        tokenizer: Tokenizer,
        placeholder_token: str,
    ) -> Tokenizer:
        num_added_tokens = tokenizer.add_tokens(placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_token}. "
                "Please pass a different `placeholder_token` that is not already in the tokenizer."
            )
        return tokenizer
