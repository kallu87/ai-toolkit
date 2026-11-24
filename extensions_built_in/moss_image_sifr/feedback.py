import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable


def write_atomic(path: Path, write: Callable) -> None:
    file = NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(file.name)
    try:
        with file:
            write(file)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    write_atomic(path, lambda file: file.write((data + "\n").encode("utf-8")))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_scores(path: Path, manifest: dict) -> list[int]:
    submission = read_json(path)
    if not isinstance(submission, dict) or submission.get("round_id") != manifest["round_id"]:
        raise ValueError("Ratings must use the round_id from this round's template")
    ids = [sample["id"] for sample in manifest["samples"]]
    scores = submission.get("scores")
    if not isinstance(scores, dict) or scores.keys() != set(ids):
        raise ValueError("Supply a rating for every sample ID in the round")
    if any(type(score) is not int or not 1 <= score <= 5 for score in scores.values()):
        raise ValueError("Each rating must be an integer from 1 to 5")
    return [scores[sample_id] for sample_id in ids]
