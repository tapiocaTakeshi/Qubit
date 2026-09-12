from pathlib import Path

import pytest

sentencepiece = pytest.importorskip("sentencepiece")

from build_tokenizer_from_vocab import build_tokenizer_model


def test_builds_a_loadable_model_without_downloading_training_data(tmp_path: Path) -> None:
    vocab_path = tmp_path / "tokenizer.vocab"
    output_path = tmp_path / "tokenizer.model"
    entries = [
        ("<pad>", 0.0),
        ("<unk>", 0.0),
        ("<s>", 0.0),
        ("</s>", 0.0),
        ("<USER>", 0.0),
        ("<ASSISTANT>", 0.0),
        ("<SYSTEM>", 0.0),
        ("<bof>", 0.0),
        ("<eof>", 0.0),
        ("▁", -0.1),
        ("テ", -0.2),
        ("ス", -0.3),
        ("ト", -0.4),
    ]
    vocab_path.write_text(
        "\n".join(f"{piece}\t{score}" for piece, score in entries) + "\n",
        encoding="utf-8",
    )

    build_tokenizer_model(vocab_path, output_path)

    processor = sentencepiece.SentencePieceProcessor(model_file=str(output_path))
    assert processor.GetPieceSize() == len(entries)
    assert [processor.IdToPiece(index) for index in range(len(entries))] == [
        piece for piece, _ in entries
    ]
    assert processor.PieceToId("<USER>") == 4
