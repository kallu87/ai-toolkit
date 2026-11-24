"""Verify two MOSS Image SIFR rounds with fixed test ratings on Modal."""

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path, PurePosixPath

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = PurePosixPath("/root/ai-toolkit")
OUTPUT = PurePosixPath("/results")
VOLUME_NAME = "moss-image-sifr-smoke-results"
MODEL_REPO = "Comfy-Org/MOSS-Image"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .uv_pip_install(
        "torch==2.13.0", "torchvision==0.28.0", "torchaudio==2.11.0",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .uv_pip_install("ftfy", "scipy==1.12.0", requirements=[str(ROOT / "requirements_base.txt")])
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .workdir(str(REMOTE_ROOT))
)
for directory in ("assets", "extensions", "extensions_built_in", "jobs", "toolkit"):
    image = image.add_local_dir(
        ROOT / directory, str(REMOTE_ROOT / directory),
        ignore=["**/__pycache__/**", "**/.env*"],
    )
for file in ("run.py", "info.py", "version.py", "config/examples/train_moss_image_sifr.yaml"):
    image = image.add_local_file(ROOT / file, str(REMOTE_ROOT / file))

app = modal.App("moss-image-sifr-smoke")
results = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


@app.function(image=image, gpu=["L40S", "A100-80GB"], memory=49152, cpu=8,
              timeout=1800, volumes={str(OUTPUT): results})
def verify(run_name: str):
    import torch
    import yaml
    from PIL import Image
    from safetensors import safe_open

    from extensions_built_in.moss_image_sifr.feedback import read_json, write_json

    assert torch.cuda.is_available(), "CUDA is required for this verification"
    gpu = torch.cuda.get_device_name()
    print(f"MOSS SIFR GPU verification on {gpu} with PyTorch {torch.__version__}")

    example = Path(REMOTE_ROOT / "config/examples/train_moss_image_sifr.yaml")
    config = yaml.safe_load(example.read_text())
    config["config"]["name"] = run_name
    process = config["config"]["process"][0]
    process["training_folder"] = str(OUTPUT)
    process["network"].update(linear=4, linear_alpha=4)
    process["train"].update(steps=2)
    process["save"].update(save_every=1)
    process["logging"]["use_ui_logger"] = False
    process["model"].update(
        name_or_path=f"{MODEL_REPO}/diffusion_models/moss_image_sft_bf16.safetensors",
        te_name_or_path=f"{MODEL_REPO}/text_encoders/qwen3.5_9b_bf16.safetensors",
        quantize=False, quantize_te=False, layer_offloading=False,
    )
    process["sifr"].update(
        prompts=process["sifr"]["prompts"][:1], rollout_every=1, samples_per_prompt=2,
        sample_steps=4,
    )
    run_dir = Path(OUTPUT / run_name)
    run_dir.mkdir()
    config_path = run_dir / "verification_config.json"
    write_json(config_path, config)
    sifr_dir = run_dir / "sifr"
    checkpoint_path = sifr_dir / "checkpoint.pt"
    active = sifr_dir / "active.json"
    reviewed = {}
    initial = None

    with subprocess.Popen([sys.executable, "run.py", str(config_path)], cwd=REMOTE_ROOT) as trainer:
        try:
            while trainer.poll() is None:
                try:
                    manifest = read_json(active)
                except FileNotFoundError:
                    time.sleep(0.5)
                    continue
                step = manifest["step"]
                if step in reviewed:
                    time.sleep(0.5)
                    continue
                if initial is None:
                    initial = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                round_dir = sifr_dir / f"round-{step:09d}"
                for sample in manifest["samples"]:
                    with Image.open(round_dir / f"{sample['id']}.png") as preview:
                        assert preview.size == (512, 512)
                        preview.verify()
                scores = {sample["id"]: score for sample, score in zip(manifest["samples"], (5, 1), strict=True)}
                write_json(round_dir / "scores.json", {"round_id": manifest["round_id"], "scores": scores})
                reviewed[step] = manifest["round_id"]
                print(f"Submitted fixed verification ratings for generated images at step {step}")
            assert trainer.returncode == 0, f"AI Toolkit exited with code {trainer.returncode}"
        finally:
            if trainer.poll() is None:
                trainer.terminate()
            results.commit()

    final = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    export = run_dir / f"{run_name}.safetensors"
    assert initial is not None and initial["step"] == 0
    assert sorted(reviewed) == [0, 1] and len(set(reviewed.values())) == 2
    assert final["step"] == 2 and final["budget"] > 0 and final["drift_ema"] > 0
    assert final["multiplier"] != initial["multiplier"]
    assert max(state["step"].item() for state in final["optimizer"]["state"].values()) == 2
    assert all(torch.isfinite(tensor).all() for tensor in final["live"].values())
    changed = sum(not torch.equal(initial["live"][key], weight) for key, weight in final["live"].items())
    assert changed > 0
    with safe_open(export, framework="pt", device="cpu") as file:
        assert file.keys()

    report = {
        "gpu": gpu, "optimizer_steps": final["step"], "generated_images": 4,
        "review_steps": sorted(reviewed), "changed_lora_tensors": changed,
        "drift_ema": final["drift_ema"], "drift_budget": final["budget"],
        "initial_lambda": initial["multiplier"], "final_lambda": final["multiplier"],
        "volume": VOLUME_NAME, "result_folder": run_name, "lora_export": export.name,
    }
    write_json(run_dir / "verification.json", report)
    results.commit()
    return report


@app.local_entrypoint()
def main():
    run_name = f"moss-sifr-{uuid.uuid4().hex[:8]}"
    print(f"Starting GPU verification with results at {VOLUME_NAME}/{run_name}")
    print(json.dumps(verify.remote(run_name), indent=2))
