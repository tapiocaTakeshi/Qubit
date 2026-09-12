"""Create a self-contained SentencePiece model from the checked-in vocabulary.

RunPod image builds do not have to be able to download Hugging Face datasets.
The vocabulary was produced together with the Qubit checkpoints, so preserving its
piece order is essential: token IDs are embedding-table indexes.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as model_pb2


REQUIRED_PIECES = {
    0: "<pad>",
    1: "<unk>",
    2: "<s>",
    3: "</s>",
}
USER_DEFINED_PIECES = {"<USER>", "<ASSISTANT>", "<SYSTEM>", "<bof>", "<eof>"}


def read_vocab(vocab_path: Path) -> list[tuple[str, float]]:
    """Read SentencePiece's TSV vocabulary without changing piece order."""
    entries: list[tuple[str, float]] = []
    seen: set[str] = set()

    for line_number, raw_line in enumerate(
        vocab_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            piece, score_text = raw_line.rsplit("\t", 1)
            score = float(score_text)
        except ValueError as exc:
            raise ValueError(
                f"{vocab_path}:{line_number} is not a SentencePiece TSV entry"
            ) from exc

        if not piece:
            raise ValueError(f"{vocab_path}:{line_number} has an empty piece")
        if not math.isfinite(score):
            raise ValueError(f"{vocab_path}:{line_number} has a non-finite score")
        if piece in seen:
            raise ValueError(f"{vocab_path}:{line_number} repeats piece {piece!r}")

        seen.add(piece)
        entries.append((piece, score))

    for index, expected_piece in REQUIRED_PIECES.items():
        if len(entries) <= index or entries[index][0] != expected_piece:
            actual_piece = entries[index][0] if len(entries) > index else None
            raise ValueError(
                f"{vocab_path} must contain {expected_piece!r} at id {index}; "
                f"found {actual_piece!r}"
            )

    if not USER_DEFINED_PIECES.issubset(seen):
        missing = ", ".join(sorted(USER_DEFINED_PIECES - seen))
        raise ValueError(f"{vocab_path} is missing required user-defined pieces: {missing}")

    return entries


def _seed_model(directory: Path) -> model_pb2.ModelProto:
    """Create a tiny local model solely to obtain SentencePiece's normalizer data."""
    corpus_path = directory / "normalizer_seed.txt"
    corpus_path.write_text(
        "日本語 English 123\n"
        "SentencePiece tokenizer normalization seed.\n"
        "会話 エージェント 関数呼び出し\n",
        encoding="utf-8",
    )
    model_prefix = directory / "normalizer_seed"
    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        model_type="unigram",
        vocab_size=64,
        character_coverage=1.0,
        hard_vocab_limit=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
    )

    model = model_pb2.ModelProto()
    model.ParseFromString((model_prefix.with_suffix(".model")).read_bytes())
    return model


def _piece_type(piece: str) -> int:
    if piece == "<unk>":
        return model_pb2.ModelProto.SentencePiece.UNKNOWN
    if piece in {"<pad>", "<s>", "</s>"}:
        return model_pb2.ModelProto.SentencePiece.CONTROL
    if piece in USER_DEFINED_PIECES:
        return model_pb2.ModelProto.SentencePiece.USER_DEFINED
    return model_pb2.ModelProto.SentencePiece.NORMAL


def build_tokenizer_model(vocab_path: Path, output_path: Path) -> None:
    """Build and validate a model whose IDs exactly match ``vocab_path``."""
    entries = read_vocab(vocab_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="qubit-tokenizer-") as temp_dir:
        model = _seed_model(Path(temp_dir))

    del model.pieces[:]
    model.trainer_spec.Clear()
    spec = model.trainer_spec
    spec.model_type = model_pb2.TrainerSpec.UNIGRAM
    spec.vocab_size = len(entries)
    spec.pad_id = 0
    spec.unk_id = 1
    spec.bos_id = 2
    spec.eos_id = 3
    spec.pad_piece = "<pad>"
    spec.unk_piece = "<unk>"
    spec.bos_piece = "<s>"
    spec.eos_piece = "</s>"
    spec.user_defined_symbols.extend(
        piece for piece, _ in entries if piece in USER_DEFINED_PIECES
    )

    for piece, score in entries:
        sentence_piece = model.pieces.add()
        sentence_piece.piece = piece
        sentence_piece.score = score
        sentence_piece.type = _piece_type(piece)

    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(model.SerializeToString())

    try:
        processor = spm.SentencePieceProcessor(model_file=str(temporary_path))
        if processor.GetPieceSize() != len(entries):
            raise RuntimeError(
                "Generated tokenizer has an unexpected vocabulary size: "
                f"{processor.GetPieceSize()} instead of {len(entries)}"
            )
        for token_id, (piece, _) in enumerate(entries):
            if processor.IdToPiece(token_id) != piece:
                raise RuntimeError(
                    f"Generated tokenizer changed id {token_id}: "
                    f"expected {piece!r}, got {processor.IdToPiece(token_id)!r}"
                )
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an offline SentencePiece model from a checked-in vocabulary."
    )
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    build_tokenizer_model(arguments.vocab, arguments.output)
    print(f"Created {arguments.output} from {arguments.vocab}")


if __name__ == "__main__":
    main()
