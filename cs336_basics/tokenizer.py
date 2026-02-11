import json
from typing import Iterable
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = set(special_tokens) if special_tokens is not None else set()
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_file(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
        with open(merges_filepath, "r") as f:
            merges = json.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # Handle special tokens if they exist
        if self.special_tokens:
            # Sort special tokens by length (longest first) for greedy matching
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "(" + "|".join(re.escape(token) for token in sorted_special_tokens) + ")"
            text_parts = [part for part in re.split(special_pattern, text) if part]
        else:
            text_parts = [text]
        
        pretokens = []
        for part in text_parts:
            if part in self.special_tokens:
                pretokens.append(part)
            else:
                for match in re.finditer(PAT, part):
                    word = match.group()
                    pretokens.append(word)
        ids = []
        for token in pretokens:
            if token in self.special_tokens:
                ids.append(self.vocab_reverse[token.encode("utf-8")])
            else:
                word_token = token.encode('utf-8')
                word_bytes = tuple(bytes([b]) for b in word_token)
                merged = self._apply_merges(word_bytes)
                for byte in merged:
                    ids.append(self.vocab_reverse[byte])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        all_bytes = [self.vocab[id] for id in ids]
        bytes_str = b"".join(all_bytes)
        decoded = bytes_str.decode("utf-8", errors="replace")
        return "".join(decoded)

    def _apply_merges(self, word_bytes: tuple[bytes, bytes]) -> tuple[bytes, bytes]:
        res = word_bytes
        for merge in self.merges:
            new_res = []
            i = 0
            while i < len(res) - 1:
                if res[i:i+2] == merge:
                    new_res.append(merge[0] + merge[1])
                    i += 2
                else:
                    new_res.append(res[i])
                    i += 1
            if i == len(res) - 1:
                new_res.append(res[i])   
            res = tuple(new_res)
        return tuple(res)