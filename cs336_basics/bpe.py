import regex as re
import multiprocessing as mp
import os
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def train_bpe(
            input_path: str, 
            vocab_size: int,
            special_tokens: list[str]
            ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input text.
    """
    # initialize the vocab
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    idx = len(vocab)
    for token in special_tokens:
        vocab[idx] = token.encode('utf-8')
        idx += 1

    # pre-tokenize the input text
    pretokens = pretokenize_parallel(input_path, special_tokens, num_processes=4)
    
    merges = []
    num_merges = vocab_size - len(vocab)
    # count pairs once, then maintain incrementally
    pair_counts = count_pairs(pretokens)
    for _ in range(num_merges):
        if not pair_counts:
            break
        # find the pair with the highest frequency
        top_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        # merge the pair and update pair_counts incrementally
        pretokens = merge_pretokens(pretokens, top_pair, pair_counts)
        # add the merged pair to the merges list
        merges.append(top_pair)
        # add the merged pair to the vocab
        vocab[idx] = top_pair[0] + top_pair[1]
        idx += 1
    return vocab, merges

def pretokenize_parallel(input_path: str, special_tokens: list[str], 
                        num_processes: int = 8) -> dict:
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
    
    tasks = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        tasks.append((input_path, start, end, special_tokens))
    
    with mp.Pool(num_processes) as pool:
        results = pool.starmap(process_chunk, tasks)
    
    pretokens = {}
    for chunk_pretokens in results:
        for token, count in chunk_pretokens.items():
            if token in pretokens:
                pretokens[token] += count
            else:
                pretokens[token] = count
    
    return pretokens

def process_chunk(file_path: str, start: int, end: int, 
                 special_tokens: list[str]) -> dict:

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    
    try:
        chunk_text = chunk_bytes.decode('utf-8').replace('\r\n', '\n')
    except UnicodeDecodeError:
        chunk_text = chunk_bytes.decode('utf-8', errors='ignore').replace('\r\n', '\n')
    
    if special_tokens:
        pattern = "|".join(re.escape(token) for token in special_tokens)
        text_parts = [part for part in re.split(pattern, chunk_text) if part]
    else:
        text_parts = [chunk_text]
    
    pretokens = {}
    for part in text_parts:
        for match in re.finditer(PAT, part):
            word = match.group()
            word_bytes = word.encode('utf-8')
            word_tuple = tuple(bytes([b]) for b in word_bytes)
            pretokens[word_tuple] = pretokens.get(word_tuple, 0) + 1
    
    return pretokens

def count_pairs(pretokens: dict) -> dict:
    pair_counts = {}
    for token, count in pretokens.items():
        for i in range(len(token) - 1):
            pair = tuple([token[i], token[i+1]])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts

def merge_pretokens(pretokens: dict, best_pair: tuple[bytes, bytes], pair_counts: dict) -> dict:
    new_pretokens = {}
    for token, count in pretokens.items():
        new_token = merge_tokens(token, best_pair)
        if new_token != token:
            # subtract old pair counts for this token
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
            # add new pair counts for the merged token
            for i in range(len(new_token) - 1):
                pair = (new_token[i], new_token[i+1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        new_pretokens[new_token] = new_pretokens.get(new_token, 0) + count
    return new_pretokens

def merge_tokens(token: tuple[bytes, bytes], best_pair: tuple[bytes, bytes]) -> tuple[bytes, bytes]:
    if len(token) < 2:
        return token
    
    new_token = []
    i = 0
    while i < len(token) - 1:
        if token[i:i+2] == best_pair:
            new_token.append(token[i] + token[i+1])
            i += 2
        else:
            new_token.append(token[i])
            i += 1
    if i == len(token) - 1:
        new_token.append(token[i])
    return tuple(new_token)