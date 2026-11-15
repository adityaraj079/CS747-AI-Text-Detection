"""Picklable adapters used by the CS747 Submission #3 notebook."""


class BPEWordTokenizer:
    """Expose a Hugging Face tokenizers BPE tokenizer to scikit-learn."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        return self.tokenizer.encode(text).tokens
