import tiktoken

class Tokenizer:
    def __init__(self, model_name="gpt2"):
        self.encoder = tiktoken.get_encoding(model_name)
        self.vocab_size = self.encoder.n_vocab

    def encode(self, text, allowed_special={'<|endoftext|>'}):
        return self.encoder.encode(text, allowed_special=allowed_special)

    def decode(self, tokens):
        return self.encoder.decode(tokens)

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def decode_batch(self, token_batches):
        return [self.decode(tokens) for tokens in token_batches]
