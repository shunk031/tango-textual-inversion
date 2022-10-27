from tango.integrations.transformers.tokenizer import Tokenizer
from transformers import CLIPTokenizer

Tokenizer.register("clip", constructor="from_pretrained")(CLIPTokenizer)
