import regex as re
import multiprocessing as mp
import os
from typing import BinaryIO
from collections import Counter, defaultdict
import string
import heapq
import json
from multiprocessing import Manager, Process, Queue
from queue import Empty

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = min(4, os.cpu_count() or 1)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None    
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_merges = vocab_size - 256 - (len(special_tokens) if special_tokens else 0)
    vocab: dict[int, bytes] = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    #  Pre-tokenization
    #  Find chunk boundaries
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f, desired_num_chunks=NUM_PROCESSES, split_special_token=b"\n"
        )


    #  Count word frequencies across chunks using multiprocessing
    manager = Manager()
    queue = manager.Queue()
    processes: list[Process] = []

    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        p = Process(
            target=pre_tokenize_string_worker,
            args=(input_path, special_tokens, queue, start, end, False),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    word_counter = Counter()
    for _ in range(len(processes)):
        try:
            partial_counter = queue.get(timeout=10)
            word_counter.update(partial_counter)
        except Empty:
            continue

    pairs_counter = Counter()
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    for word in word_counter:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_to_words[pair].add(word)
            pairs_counter[pair] += word_counter[word]
    """
    word_counter is like:
    {(104, 101, 201, 1, 2)): number of times}
    The tuple is the byte representation of the word.
    pairs_counter is like:
    {(104, 101): number of times}
    pair_to_words is like:
    {(104, 101): {(104, 101, 201, 1, 2)}}
    vocab is like:
    {0: b'a'}
    """
    # BPE Core Loop
    pair_heap = build_pair_heap(pairs_counter, vocab)

    for i in range(num_merges):
        most_frequent_pair = pop_most_frequent_pair(pair_heap, pairs_counter)
        new_id = update_vocab(vocab, most_frequent_pair)

        word_counter, pairs_counter, pair_heap, pair_to_words = merge_pairs_with_heap_index(
            word_counter, pairs_counter, most_frequent_pair, new_id, vocab, pair_heap, pair_to_words
        )

        merges.append((vocab[most_frequent_pair[0]], vocab[most_frequent_pair[1]]))


    return vocab, merges

def init_vocab(special_tokens: list[str] | None = None) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # idx -> byte representation
    current_index = 256

    if special_tokens:
        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            vocab[current_index] = token_bytes
            current_index += 1

    return vocab

def pair_counts(word_counter: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    pairs: dict[tuple[int, int], int] = {}

    for word, count in word_counter.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] = pairs.get((a, b), 0) + count

    return pairs

def get_most_frequent_pair(
    pair_counter: dict[tuple[int, int], int],
) -> tuple[int, int]:
    max_freq = max(pair_counter.values())
    candidates = [pair for pair, freq in pair_counter.items() if freq == max_freq]
    res = max(candidates)

    return res

def add_pair_to_vocab(
    vocab: dict[int, bytes],
    pair: tuple[int, int],
) -> int:
    index1, index2 = pair
    vocab[len(vocab)] = vocab[index1] + vocab[index2]
    return len(vocab) - 1

def merge_pair_ids(
    word_counter: dict[tuple[bytes] | tuple[int], int],
    pair: tuple[int, int],
    new_id: int,
) -> tuple[dict[tuple[int], int], dict[tuple[int, int], int]]:
    new_word_counter: defaultdict[tuple[int], int] = defaultdict(int)
    updated_pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

    for token, freq in word_counter.items():
        new_token = []
        i = 0
        L = len(token)

        while i < L:
            if i + 1 < L and (token[i], token[i + 1]) == pair:
                new_token.append(new_id)
                i += 2
            else:
                new_token.append(token[i])
                i += 1

        new_word_counter[tuple(new_token)] += freq

        for index1, index2 in zip(new_token[:-1], new_token[1:]):
            updated_pair_counts[(index1, index2)] += freq

    return dict(new_word_counter), dict(updated_pair_counts)    

def string_to_bytes(text: str, return_int: bool = False):
    text_bytes = text.encode("utf-8")
    if return_int:
        return [b for b in text_bytes]
    return text_bytes

def split_by_special_tokens(text: str, special_tokens: list[str], include_special: bool = False) -> list[str]:
    if not special_tokens:
        return [text]

    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in special_tokens_sorted)

    if include_special:
        special_chunks = re.split(f"({pattern})", text)
    else:
        # Split without capturing the special tokens
        special_chunks = re.split(pattern, text)

    return special_chunks

def pre_tokenize(string: str, special_tokens: list[str], including_special: bool = False) -> Counter:
    word_counter = Counter()

    chunks = split_by_special_tokens(string, special_tokens, include_special=including_special)

    for chunk in chunks:
        if including_special and chunk in special_tokens:
            word_counter[tuple(string_to_bytes(chunk))] += 1
        else:
            for match in re.finditer(PAT, chunk):
                word = match.group(0)
                word_encoded = tuple(string_to_bytes(word, return_int=True))
                word_counter[word_encoded] += 1

    return word_counter

def pre_tokenize_string_worker(*args):
    input_path, special_tokens, queue, start, end, include_special = args

    # Read the chunk from the file
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    # Normalize line endings
    chunk = chunk.replace('\r\n', '\n').replace('\r', '\n')

    word_counter = pre_tokenize(chunk, special_tokens, include_special)

    # Put the result in the queue
    queue.put(word_counter)

class HeapItem:
    def __init__(self, neg_freq: int, pair_bytes: tuple[bytes, bytes], pair: tuple[int, int]):
        self.neg_freq = neg_freq
        self.pair_bytes = pair_bytes
        self.pair = pair

    def __lt__(self, other: "HeapItem") -> bool:
        if self.neg_freq != other.neg_freq:
            return self.neg_freq < other.neg_freq
        return self.pair_bytes > other.pair_bytes  # tie-break by lexicographic order


def build_pair_heap(pairs_freqs: Counter, vocab: dict[int, bytes]):
    heap = []
    for (a, b), f in pairs_freqs.items():
        if f > 0:
            item = HeapItem(-f, (vocab[a], vocab[b]), (a, b))
            heapq.heappush(heap, item)
    return heap


def pop_most_frequent_pair(heap, pairs_counter: Counter) -> tuple[int, int]:
    while heap:
        item = heapq.heappop(heap)  # Pop the top item
        neg_f = item.neg_freq
        pair = item.pair
        cur_f = pairs_counter.get(pair, 0)
        if cur_f <= 0 or -neg_f != cur_f:  # frequency changed, which means the pair we store in heap is stale
            continue
        return pair

    raise ValueError("No positive-frequency pairs remain")


def merge_pairs_with_heap_index(
    word_counter: dict[tuple[int, ...], int],
    pair_counter: Counter,
    target_pair: tuple[int, int],
    new_id: int,
    vocab: dict[int, bytes],
    pair_heap,
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]],
) -> tuple[
    dict[tuple[int, ...], int],
    Counter,
    list,
    dict[tuple[int, int], set[tuple[int, ...]]],
]:
    # Start from full counters so unaffected words remain.
    new_word_counter: Counter = Counter(word_counter)
    updated_pair_counter: Counter = pair_counter.copy()
    changed_pairs: set[tuple[int, int]] = set()

    # Get all words that contain the target pair.
    affected_words = list(pair_to_words.get(target_pair, set()))

    for w in affected_words:
        freq = word_counter.get(w, 0)
        if freq <= 0 or len(w) < 2:
            continue

        # Remove the old word from the corpus counts.
        new_word_counter[w] -= freq
        if new_word_counter[w] <= 0:
            del new_word_counter[w]

        # Subtract ALL old adjacent pairs for this word + remove old word from index.
        for i in range(len(w) - 1):
            pair = (w[i], w[i + 1])
            updated_pair_counter[pair] -= freq
            changed_pairs.add(pair)

            s = pair_to_words.get(pair)
            if s is not None:
                s.discard(w)
                if not s:
                    del pair_to_words[pair]

        # Build merged word (greedy left-to-right, same as standard BPE).
        new_word = get_new_word(w, target_pair, new_id)
        new_word_counter[new_word] += freq

        # Add ALL new adjacent pairs for merged word + add merged word into index.
        if len(new_word) >= 2:
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                updated_pair_counter[pair] += freq
                changed_pairs.add(pair)
                pair_to_words.setdefault(pair, set()).add(new_word)

    # Push updated frequencies for changed pairs into heap (skip non-positive).
    if pair_heap is not None:
        for p in changed_pairs:
            f = updated_pair_counter.get(p, 0)
            if f > 0:
                heapq.heappush(pair_heap, HeapItem(-f, (vocab[p[0]], vocab[p[1]]), p))

    return dict(new_word_counter), updated_pair_counter, pair_heap, pair_to_words

def get_new_word(
    word: tuple[int, ...],
    target_pair: tuple[int, int],
    new_id: int,
) -> tuple[int, ...]:
    a, b = target_pair
    new_word = []
    i = 0

    while i < len(word):
        if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1

    return tuple(new_word)


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

def update_vocab(vocab: dict[int, bytes], pair: tuple[int, int]) -> int:
    new_id = len(vocab)
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return new_id