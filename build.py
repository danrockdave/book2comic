"""Generate the voices and assemble the narrated video of a story.

  python build.py voices
      Writes a voice sample per character to out/voices/ to tweak characters.json.

  python build.py story stories/02_o_bruxo.json
      Voices every line (edge-tts), joins them with the images from images/02/
      and writes out/02_o_bruxo.mp4 (with toggleable subtitles) plus
      out/02/timeline.json. Missing images become placeholders, so you can
      build before generating them.

Options for 'story':
  --keep-tags   also narrate short dialogue tags ("disse Velerad")
  --dry-run     no internet: silent audio sized to the text (pipeline test)
  --size 1280x720
"""

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import subprocess
import textwrap
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import ROOT, load_characters, load_story, story_slug

SAMPLE_RATE = 24000
FPS = 25
PAUSE_SEGMENT = 0.25
PAUSE_PARAGRAPH = 0.55
MIN_FRAME_SECONDS = 2.5
FADE = 0.35
TTS_CONCURRENCY = 4
VOICE_SAMPLE = "Sou um bruxo de Rívia. Não estou interessado na princesa; o importante são os três mil ducados."


# ---------------------------------------------------------------- timeline

def is_dialogue_tag(paragraph, index, segment):
    """Short narration inside a dialogue paragraph, e.g. 'disse Velerad.'"""
    first = paragraph["segments"][0]
    text = segment["text"]
    return (index > 0 and first["kind"] == "speech" and segment["kind"] == "narration"
            and len(text) < 90 and text[:1].islower())


def build_timeline(annotation, story, keep_tags):
    speaker_of = {}
    for char_id, ids in annotation["speakers"].items():
        for pid in ids.split():
            speaker_of[f"p{pid}"] = char_id
    fixes = annotation.get("text_fixes", {})
    starts = {f["start"]: f for f in annotation["frames"]}

    frames = []
    current = None
    for section in story["sections"]:
        for paragraph in section["paragraphs"]:
            if paragraph["id"] in starts:
                current = {**starts[paragraph["id"]], "lines": []}
                frames.append(current)
            if current is None:
                raise SystemExit("O primeiro quadro precisa começar no primeiro parágrafo do conto.")
            first_line_of_paragraph = True
            for i, segment in enumerate(paragraph["segments"]):
                if not keep_tags and is_dialogue_tag(paragraph, i, segment):
                    continue
                text = segment["text"]
                for old, new in fixes.get(segment["id"], []):
                    text = text.replace(old, new)
                if segment["kind"] == "speech":
                    speaker = speaker_of.get(paragraph["id"], "desconhecido")
                else:
                    speaker = "narrador"
                current["lines"].append({
                    "id": segment["id"], "speaker": speaker, "text": text,
                    "pause_before": PAUSE_PARAGRAPH if first_line_of_paragraph else PAUSE_SEGMENT,
                })
                first_line_of_paragraph = False
    return frames


# ---------------------------------------------------------------- audio

def voice_settings(characters, speaker):
    char = characters.get(speaker) or characters["desconhecido"]
    if "voice" not in char:
        char = characters["desconhecido"]
    return char["voice"], char.get("rate", "+0%"), char.get("pitch", "+0Hz")


def audio_key(text, voice, rate, pitch):
    return hashlib.sha1(f"{voice}|{rate}|{pitch}|{text}".encode()).hexdigest()[:12]


async def tts_one(text, voice, rate, pitch, out_path, semaphore):
    import edge_tts
    async with semaphore:
        for attempt in range(4):
            try:
                tmp = out_path.with_suffix(".part")
                await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(tmp))
                tmp.replace(out_path)
                return
            except Exception as exc:  # network hiccups are common on long runs
                if attempt == 3:
                    raise RuntimeError(f"Falha no TTS ({voice}): {text[:60]}") from exc
                await asyncio.sleep(2 * (attempt + 1))


async def tts_all(jobs):
    semaphore = asyncio.Semaphore(TTS_CONCURRENCY)
    tasks = [tts_one(*job, semaphore) for job in jobs]
    done = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        done += 1
        if done % 25 == 0 or done == len(tasks):
            print(f"  voz {done}/{len(tasks)}")


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def to_wav(src, dst):
    run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-sample_fmt", "s16", str(dst)])


def write_silence_wav(path, seconds):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * int(seconds * SAMPLE_RATE))


def read_pcm(path):
    with wave.open(str(path), "rb") as w:
        return w.readframes(w.getnframes())


def synthesize(frames, characters, audio_dir, dry_run):
    """Fill line['wav'] for every line, generating only what is not cached."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for frame in frames:
        for line in frame["lines"]:
            voice, rate, pitch = voice_settings(characters, line["speaker"])
            key = audio_key(line["text"], voice, rate, pitch)
            line["mp3"] = audio_dir / f"{line['id']}_{key}.mp3"
            line["wav"] = audio_dir / f"{line['id']}_{key}.wav"
            if line["wav"].exists():
                continue
            if dry_run:
                write_silence_wav(line["wav"], max(0.6, len(line["text"]) / 15))
            elif not line["mp3"].exists():
                jobs.append((line["text"], voice, rate, pitch, line["mp3"]))
    if jobs:
        print(f"Gerando {len(jobs)} falas com edge-tts...")
        asyncio.run(tts_all(jobs))
    pending = [l for f in frames for l in f["lines"] if not l["wav"].exists()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda l: to_wav(l["mp3"], l["wav"]), pending))


def assemble_frame_audio(frame, out_path, clock):
    """Concatenate the frame lines with pauses; return duration and subtitle cues."""
    pcm = bytearray()
    cues = []
    for i, line in enumerate(frame["lines"]):
        pause = 0.3 if i == 0 else line["pause_before"]
        pcm += b"\x00\x00" * int(pause * SAMPLE_RATE)
        start = clock + len(pcm) / 2 / SAMPLE_RATE
        pcm += read_pcm(line["wav"])
        end = clock + len(pcm) / 2 / SAMPLE_RATE
        cues.append((start, end, line))
    pcm += b"\x00\x00" * int(0.4 * SAMPLE_RATE)
    min_len = int(MIN_FRAME_SECONDS * SAMPLE_RATE) * 2
    if len(pcm) < min_len:
        pcm += b"\x00" * (min_len - len(pcm))
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(pcm))
    return len(pcm) / 2 / SAMPLE_RATE, cues


# ---------------------------------------------------------------- video

def pick_encoder():
    try:
        run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.2",
             "-c:v", "h264_nvenc", "-f", "null", "-"])
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23", "-b:v", "0"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]


def placeholder(frame, size, path):
    image = Image.new("RGB", size, (24, 22, 28))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
        small = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = small = ImageFont.load_default()
    # Keep the text near the center so the Ken Burns zoom does not crop it.
    x = size[0] // 6
    y = size[1] // 3
    draw.text((x, y), f"{frame['id']}  (imagem ainda não gerada)", fill=(230, 200, 120), font=font)
    y += 60
    for row in textwrap.wrap(frame["shot"], 60):
        draw.text((x, y), row, fill=(200, 200, 200), font=small)
        y += 32
    image.save(path)


def ken_burns_filter(index, duration, size):
    w, h = size
    total = max(int(duration * FPS), 1)
    progress = f"(on/{total})"
    zoom_in = f"1+0.12*{progress}"
    zoom_out = f"1.12-0.12*{progress}"
    center_x = "iw/2-(iw/zoom/2)"
    center_y = "ih/2-(ih/zoom/2)"
    moves = [
        (zoom_in, center_x, center_y),
        (zoom_out, center_x, center_y),
        ("1.12", f"(iw-iw/zoom)*{progress}", center_y),
        ("1.12", f"(iw-iw/zoom)*(1-{progress})", center_y),
    ]
    z, x, y = moves[index % len(moves)]
    fade_out = max(duration - FADE, 0)
    return (f"scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},"
            f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={w}x{h}:fps={FPS},"
            f"fade=t=in:st=0:d={FADE},fade=t=out:st={fade_out:.2f}:d={FADE},format=yuv420p")


def render_clip(index, image_path, audio_path, duration, size, encoder, out_path):
    run(["ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(image_path),
         "-i", str(audio_path), "-t", f"{duration:.3f}",
         "-vf", ken_burns_filter(index, duration, size), *encoder,
         "-c:a", "aac", "-b:a", "96k", "-ar", "48000", "-ac", "1",
         "-shortest", str(out_path)])


def srt_time(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues, characters, path):
    with open(path, "w", encoding="utf-8") as f:
        for n, (start, end, line) in enumerate(cues, 1):
            text = line["text"]
            if line["speaker"] != "narrador":
                name = characters.get(line["speaker"], {}).get("name", line["speaker"])
                text = f"{name}: {text}"
            f.write(f"{n}\n{srt_time(start)} --> {srt_time(end)}\n{textwrap.fill(text, 70)}\n\n")


def slugify(text):
    text = text.lower()
    for a, b in zip("áàâãéêíóôõúç", "aaaaeeiooouc"):
        text = text.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def cmd_story(args):
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg não encontrado no PATH.")
    size = tuple(int(v) for v in args.size.lower().split("x"))
    characters = load_characters()
    annotation, story = load_story(args.story)
    slug = story_slug(annotation)
    work = ROOT / "out" / slug
    (work / "clips").mkdir(parents=True, exist_ok=True)
    (work / "frame_audio").mkdir(exist_ok=True)
    image_dir = ROOT / "images" / slug

    frames = build_timeline(annotation, story, args.keep_tags)
    total_lines = sum(len(f["lines"]) for f in frames)
    print(f"{annotation['title']}: {len(frames)} quadros, {total_lines} falas")

    synthesize(frames, characters, work / "audio", args.dry_run)

    clock = 0.0
    cues = []
    for frame in frames:
        audio_path = work / "frame_audio" / f"{frame['id']}.wav"
        frame["duration"], frame_cues = assemble_frame_audio(frame, audio_path, clock)
        frame["audio"] = audio_path
        frame["t0"] = clock
        cues += frame_cues
        clock += frame["duration"]
    print(f"Duração total: {clock / 60:.1f} min")

    encoder = pick_encoder()
    print(f"Encoder de vídeo: {encoder[1]}")

    def render(item):
        index, frame = item
        image_path = image_dir / f"{frame['id']}.png"
        if not image_path.exists():
            image_path = work / "clips" / f"{frame['id']}_placeholder.png"
            placeholder(frame, size, image_path)
        clip = work / "clips" / f"{frame['id']}.mp4"
        stamp = clip.with_suffix(".stamp")
        signature = f"{image_path.stat().st_mtime}|{frame['duration']:.3f}|{size}"
        if clip.exists() and stamp.exists() and stamp.read_text() == signature:
            return
        render_clip(index, image_path, frame["audio"], frame["duration"], size, encoder, clip)
        stamp.write_text(signature)

    workers = 2 if encoder[1] == "h264_nvenc" else 3
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, _ in enumerate(pool.map(render, enumerate(frames)), 1):
            if n % 10 == 0 or n == len(frames):
                print(f"  quadro {n}/{len(frames)}")

    concat_list = work / "concat.txt"
    concat_list.write_text("".join(f"file 'clips/{f['id']}.mp4'\n" for f in frames), encoding="utf-8")
    srt_path = work / "legendas.srt"
    write_srt(cues, characters, srt_path)

    output = ROOT / "out" / f"{slug}_{slugify(annotation['title'])}.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-i", str(srt_path),
         "-map", "0:v", "-map", "0:a", "-map", "1:s", "-c:v", "copy", "-c:a", "copy",
         "-c:s", "mov_text", "-metadata:s:s:0", "language=por",
         "-metadata", f"title={annotation['title']}", "-movflags", "+faststart", str(output)])

    timeline = [{
        "id": f["id"], "image": f"images/{slug}/{f['id']}.png", "start": round(f["t0"], 3),
        "duration": round(f["duration"], 3),
        "lines": [{"speaker": l["speaker"], "text": l["text"]} for l in f["lines"]],
    } for f in frames]
    (work / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nPronto: {output}")


def cmd_voices(args):
    characters = load_characters()
    out_dir = ROOT / "out" / "voices"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for char_id, char in characters.items():
        if "voice" in char:
            jobs.append((VOICE_SAMPLE, char["voice"], char.get("rate", "+0%"),
                         char.get("pitch", "+0Hz"), out_dir / f"{char_id}.mp3"))
    asyncio.run(tts_all(jobs))
    print(f"Amostras em {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    story = sub.add_parser("story")
    story.add_argument("story")
    story.add_argument("--keep-tags", action="store_true")
    story.add_argument("--dry-run", action="store_true")
    story.add_argument("--size", default="1280x720")
    sub.add_parser("voices")
    args = parser.parse_args()
    {"story": cmd_story, "voices": cmd_voices}[args.cmd](args)


if __name__ == "__main__":
    main()
