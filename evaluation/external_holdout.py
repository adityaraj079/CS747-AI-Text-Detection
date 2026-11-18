"""Screen paired external examples before scoring any frozen model."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"\w+")
WINDOW = 32


def screen_pairs(samples, corpus):
    anchors = defaultdict(dict)
    excluded = set()
    seen = {}
    for number, _, _, _, text in samples:
        words = TOKEN.findall(text.casefold())
        if len(words) < WINDOW:
            excluded.add(number)
        signature = " ".join(words)
        if signature in seen:
            excluded.update((number, seen[signature]))
        seen[signature] = number
        for i in range(0, len(words) - WINDOW + 1, WINDOW):
            phrase = tuple(words[i:i + WINDOW])
            anchors[phrase[0]].setdefault(phrase, set()).add(number)

    with corpus.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        count = 0
        for row in reader:
            count += 1
            words = TOKEN.findall(row["text"].casefold())
            for i, word in enumerate(words):
                candidates = anchors.get(word)
                if candidates:
                    excluded.update(candidates.get(tuple(words[i:i + WINDOW]), ()))
    expected = json.loads((ROOT / "data/manifests/data_quality_report.json").read_text())
    if count != expected["original_row_count"]:
        raise ValueError("Use the full CS747 source corpus for the overlap check.")
    return [row for row in samples if row[0] not in excluded], sorted(excluded), count
