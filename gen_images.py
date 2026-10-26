"""Generate the frame images with SDXL running locally (diffusers).

Steps:
  1) python gen_images.py refs
     Generates 4 portrait candidates per character in refs/. Pick the best one
     and rename it to refs/<character>.png (e.g. refs/geralt.png).
  2) python gen_images.py frames stories/02_o_bruxo.json
     Generates images/02/f001.png ... Existing images are skipped, so to redo
     a bad frame just delete it (or use --only f007 f012 --seed 123) and rerun.

With refs/<character>.png present, the first character of each frame that has
a reference is passed to IP-Adapter to keep the face consistent.

Also writes images/02/prompts.txt, handy to redo a frame manually in Fooocus.
"""

import argparse
import random
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from common import (NEGATIVE, ROOT, STYLE, build_prompt, load_characters, load_story,
                    prompt_too_long, story_slug)

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
FRAME_SIZE = (1344, 768)
REF_SIZE = (896, 1152)
REF_DIR = ROOT / "refs"


def load_pipeline(model, use_ip_adapter):
    dtype = torch.float16
    image_encoder = None
    if use_ip_adapter:
        from transformers import CLIPVisionModelWithProjection
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=dtype)
    kwargs = {"torch_dtype": dtype}
    if image_encoder is not None:
        kwargs["image_encoder"] = image_encoder
    if model.endswith(".safetensors"):
        pipe = StableDiffusionXLPipeline.from_single_file(model, **kwargs)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model, variant="fp16", use_safetensors=True, **kwargs)
    if use_ip_adapter:
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                             weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors")
    # 8 GB of VRAM: keep only the active sub-model on the GPU.
    pipe.enable_model_cpu_offload()

    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    return pipe


def generate(pipe, prompt, size, seed, steps, ip_image=None, ip_scale=0.0, use_ip=False):
    generator = torch.Generator("cpu").manual_seed(seed)
    kwargs = {}
    if use_ip:
        pipe.set_ip_adapter_scale(ip_scale if ip_image is not None else 0.0)
        kwargs["ip_adapter_image"] = ip_image or Image.new("RGB", (224, 224))
    result = pipe(prompt=prompt, negative_prompt=NEGATIVE, width=size[0], height=size[1],
                  num_inference_steps=steps, guidance_scale=6.0, generator=generator, **kwargs)
    return result.images[0]


def cmd_refs(args):
    characters = load_characters()
    REF_DIR.mkdir(exist_ok=True)
    pipe = load_pipeline(args.model, use_ip_adapter=False)
    targets = args.only or [k for k, v in characters.items() if v.get("look")]
    for char_id in targets:
        look = characters[char_id]["look"]
        prompt = f"{STYLE}, character portrait, upper body, facing the viewer, plain dark background, {look}"
        for i in range(args.count):
            out = REF_DIR / f"{char_id}_{i + 1}.png"
            if out.exists():
                continue
            seed = random.randint(0, 2**31 - 1)
            print(f"[ref] {char_id} #{i + 1} (seed {seed})")
            generate(pipe, prompt, REF_SIZE, seed, args.steps).save(out)
    print(f"\nPronto. Escolha o melhor de cada um e renomeie para {REF_DIR}/<personagem>.png")


def cmd_frames(args):
    characters = load_characters()
    annotation, _ = load_story(args.story)
    out_dir = ROOT / "images" / story_slug(annotation)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = annotation["frames"]
    with open(out_dir / "prompts.txt", "w", encoding="utf-8") as f:
        for frame in frames:
            prompt = build_prompt(frame, annotation, characters)
            if prompt_too_long(prompt):
                print(f"Aviso: prompt de {frame['id']} é longo; o fim pode ser ignorado pelo SDXL")
            f.write(f"{frame['id']}: {prompt}\n")

    if args.only:
        frames = [fr for fr in frames if fr["id"] in args.only]
    todo = [fr for fr in frames if args.only or not (out_dir / f"{fr['id']}.png").exists()]
    if not todo:
        print("Todas as imagens já existem. Apague as que quiser refazer.")
        return

    refs = {p.stem: p for p in REF_DIR.glob("*.png") if "_" not in p.stem} if REF_DIR.exists() else {}
    use_ip = bool(refs) and not args.no_ip_adapter
    pipe = load_pipeline(args.model, use_ip_adapter=use_ip)
    ref_images = {k: Image.open(v).convert("RGB") for k, v in refs.items()}

    for n, frame in enumerate(todo, 1):
        prompt = build_prompt(frame, annotation, characters)
        # IP-Adapter applies the reference face to every face in the image, so it
        # is only used when the frame shows a single character.
        chars = frame.get("chars", [])
        ip_char = chars[0] if len(chars) == 1 and chars[0] in ref_images else None
        seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
        print(f"[{n}/{len(todo)}] {frame['id']} seed={seed} ref={ip_char or '-'}")
        image = generate(pipe, prompt, FRAME_SIZE, seed, args.steps,
                         ip_image=ref_images.get(ip_char), ip_scale=args.ip_scale, use_ip=use_ip)
        image.save(out_dir / f"{frame['id']}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Modelo do Hugging Face ou caminho de um .safetensors SDXL (ex.: baixado do Civitai)")
    parser.add_argument("--steps", type=int, default=28)
    sub = parser.add_subparsers(dest="cmd", required=True)

    refs = sub.add_parser("refs", help="Gera retratos de referência dos personagens")
    refs.add_argument("--count", type=int, default=4)
    refs.add_argument("--only", nargs="*", help="Só estes personagens (ex.: geralt foltest)")

    frames = sub.add_parser("frames", help="Gera as imagens dos quadros de um conto")
    frames.add_argument("story", help="Arquivo do roteiro, ex.: stories/02_o_bruxo.json")
    frames.add_argument("--only", nargs="*", help="Só estes quadros (ex.: f007 f012); sobrescreve")
    frames.add_argument("--seed", type=int)
    frames.add_argument("--ip-scale", type=float, default=0.45,
                        help="Força da referência de rosto (0.3 a 0.7)")
    frames.add_argument("--no-ip-adapter", action="store_true")

    args = parser.parse_args()
    {"refs": cmd_refs, "frames": cmd_frames}[args.cmd](args)


if __name__ == "__main__":
    main()