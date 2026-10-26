"""Shared helpers: loading the book, characters and story annotations."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STYLE = "dark fantasy graphic novel art, ink and watercolor, dramatic lighting"
MAX_PROMPT_WORDS = 70  # CLIP reads ~77 tokens; the rest is silently dropped
NEGATIVE = ("photo, photorealistic, 3d render, nsfw, nude, gore, text, letters, "
            "watermark, signature, blurry, lowres, deformed, bad anatomy, "
            "extra fingers, extra limbs, modern clothing")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_characters():
    data = load_json(ROOT / "characters.json")
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_story(story_path, book_path=ROOT / "book.json"):
    """Return (annotation, story_from_book)."""
    annotation = load_json(story_path)
    book = load_json(book_path)
    story = book["stories"][annotation["story_index"] - 1]
    if story["title"] != annotation["title"]:
        raise SystemExit(
            f"O book.json não bate com o roteiro: esperado '{annotation['title']}', "
            f"encontrado '{story['title']}'. Gere o book.json com o extract.py a partir do mesmo PDF.")
    return annotation, story


def story_slug(annotation):
    return f"{annotation['story_index']:02d}"


def build_prompt(frame, annotation, characters):
    """Compose the SDXL prompt. Order matters: CLIP keeps only ~77 tokens."""
    parts = [STYLE, frame["shot"]]
    for char_id in frame.get("chars", []):
        char = characters.get(char_id, {})
        tag = char.get("tag") or char.get("look")
        if tag:
            parts.append(tag)
    place = annotation.get("places", {}).get(frame.get("place", ""))
    if place:
        parts.append(place)
    return ", ".join(parts)


def prompt_too_long(prompt):
    return len(prompt.replace(",", " ").split()) > MAX_PROMPT_WORDS
