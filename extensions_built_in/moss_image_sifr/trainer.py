import hashlib
import json
import math
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.basic import flush
from toolkit.memory_management import sync_grad_transfers
from toolkit.print import print_acc

from .feedback import read_json, read_scores, write_atomic, write_json
from .objective import flow_batch, signed_advantages, update_multiplier, velocity_mse


@dataclass
class SIFRConfig:
    prompts: list[str]
    rollout_every: int = 8
    samples_per_prompt: int = 4
    warmup_steps: int = 0
    width: int = 512
    height: int = 512
    sample_steps: int = 20
    guidance_scale: float = 3.0
    negative_prompt: str = ""
    seed: int = 42
    rollout_policy: str = "ema"
    reference_decay: float = 0.5
    drift_ema_decay: float = 0.9
    budget_ratio: float = 0.01
    lambda_init: float = 0.01
    lambda_min: float = 0.001
    lambda_max: float = 10.0
    lambda_rate: float = 0.1

    def __post_init__(self):
        if not isinstance(self.prompts, list) or not self.prompts or any(
            not isinstance(prompt, str) or not prompt.strip() for prompt in self.prompts
        ):
            raise ValueError("sifr.prompts must contain one or more nonempty prompt strings")
        if any(type(value) is not int or value < minimum for value, minimum in (
            (self.rollout_every, 1), (self.samples_per_prompt, 2), (self.warmup_steps, 0),
            (self.width, 16), (self.height, 16), (self.sample_steps, 1), (self.seed, 0),
        )):
            raise ValueError("SIFR step counts, seed, dimensions and group size must be valid integers")
        if self.width % 16 or self.height % 16:
            raise ValueError("SIFR image dimensions must be multiples of 16")
        if self.rollout_policy not in ("ema", "live"):
            raise ValueError("sifr.rollout_policy must be 'ema' or 'live'")
        if not all(0 <= decay < 1 for decay in (self.reference_decay, self.drift_ema_decay)):
            raise ValueError("SIFR EMA decays must be between zero and one, excluding one")
        if not all(math.isfinite(value) and value > 0 for value in (
            self.budget_ratio, self.lambda_init, self.lambda_min, self.lambda_max, self.lambda_rate,
        )) or not self.lambda_min <= self.lambda_init <= self.lambda_max:
            raise ValueError("SIFR budget ratio and multiplier settings must be positive and finite, with lambda_init within bounds")
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 1:
            raise ValueError("sifr.guidance_scale must be at least 1")


class MossImageSIFRTrainer(DiffusionTrainer):
    """Alternates MOSS Image rollouts, human ratings, and blocks of SIFR updates."""

    def __init__(self, process_id, job, config, **kwargs):
        sifr = config.get("sifr")
        if not isinstance(sifr, dict) or "prompts" not in sifr:
            raise ValueError("The moss_image_sifr trainer requires a sifr.prompts list")
        self.sifr = SIFRConfig(**sifr)
        super().__init__(process_id, job, config, **kwargs)
        train = self.train_config
        model = self.model_config
        network = self.network_config
        if model.arch != "moss_image" or model.model_kwargs.get("edit"):
            raise ValueError("SIFR requires the MOSS Image text-to-image architecture")
        if (network is None or network.type != "lora" or network.all_layers
                or train.train_text_encoder or not train.train_unet
                or self.adapter_config or self.embed_config or self.decorator_config):
            raise ValueError("SIFR requires a transformer LoRA and a frozen text encoder")
        if (self.accelerator.num_processes != 1 or train.gradient_accumulation != 1
                or train.gradient_accumulation_steps != 1 or train.do_paramiter_swapping):
            raise ValueError("SIFR requires a single training process and gradient accumulation of 1")
        if (train.merge_network_on_save or train.ema_config.use_ema or train.learnable_snr_gos
                or model.assistant_lora_path or model.inference_lora_path):
            raise ValueError("SIFR manages its own LoRA reference and requires unmerged LoRA checkpoints")
        if train.steps <= self.sifr.warmup_steps or train.batch_size < 1:
            raise ValueError("SIFR needs a positive batch size and training steps beyond warmup")
        if self.sifr.warmup_steps and not self.datasets:
            raise ValueError("Set datasets for supervised SIFR warmup, or set warmup_steps to 0")

        self.feedback_root = Path(self.save_root) / "sifr"
        self.state_path = self.feedback_root / "checkpoint.pt"
        self.active_path = self.feedback_root / "active.json"
        self.run_id = uuid.uuid4().hex
        self.committed_step = None
        self.round_dir = None
        self.rollout_latents = None
        self.budget = self.drift_ema = None
        self.multiplier = self.sifr.lambda_init
        self.last_export_step = 0
        # The base trainer's interval saves run before it advances step_num.
        # Save from end_step_hook so the LoRA metadata records completed updates.
        self.checkpoint_every = self.save_config.save_every
        self.save_config.save_every = 0

    def needs_training_dataset(self, step: int) -> bool:
        # The SIFR checkpoint takes precedence over a potentially older LoRA
        # export. It is restored after the dataset setup stage.
        return step < self.sifr.warmup_steps and not self.state_path.is_file()

    @torch.no_grad()
    def before_dataset_load(self):
        super().before_dataset_load()
        self.prompts = [self.sd.inject_trigger_into_prompt(prompt, self.trigger_word) for prompt in self.sifr.prompts]
        self.prompt_embeds = [self.sd.get_prompt_embeds(prompt).to("cpu") for prompt in self.prompts]
        self.negative_embeds = None
        if self.sifr.guidance_scale != 1:
            self.negative_embeds = self.sd.get_prompt_embeds(self.sifr.negative_prompt).to("cpu")
        if self.model_config.low_vram:
            self.sd.text_encoder_to("cpu")
            flush()

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        self.feedback_root.mkdir(parents=True, exist_ok=True)
        trainable = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        self.lora_parameters = {name: p for name, p in self.network.named_parameters() if id(p) in trainable}
        if not self.lora_parameters or len(self.lora_parameters) != len(trainable):
            raise RuntimeError("SIFR optimizer must contain only LoRA parameters")
        identity = [asdict(self.sifr), self.prompts, self.config["model"], self.config["network"],
                    self.train_config.batch_size, self.train_config.optimizer, self.train_config.lr_scheduler]
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.last_export_step = self.step_num
        if self.state_path.exists():
            state = torch.load(self.state_path, map_location="cpu", weights_only=True)
            if state["fingerprint"] != self.fingerprint:
                raise ValueError("SIFR settings changed for this checkpoint. Use a new job name to start another run")
            self._copy_lora(state["live"])
            self.reference = state["reference"]
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr_scheduler.load_state_dict(state["scheduler"])
            if self.accelerator.scaler is not None:
                self.accelerator.scaler.load_state_dict(state["scaler"])
            self.budget, self.drift_ema, self.multiplier = state["budget"], state["drift_ema"], state["multiplier"]
            self.run_id = state["run_id"]
            self.committed_step = self.step_num = self.start_step = self.last_save_step = state["step"]
            self.update_step()
            print_acc(f"Resumed SIFR at optimizer step {self.step_num}")
        if self.step_num > self.train_config.steps:
            raise ValueError("train.steps is below the saved SIFR step")
        if self.active_path.exists():
            active = read_json(self.active_path)
            if (active["run_id"], active["step"]) != (self.run_id, self.step_num):
                self.active_path.unlink()

    def _snapshot_lora(self):
        return {name: p.detach().to(device="cpu", copy=True) for name, p in self.lora_parameters.items()}

    @torch.no_grad()
    def _copy_lora(self, weights):
        for name, parameter in self.lora_parameters.items():
            parameter.copy_(weights[name])

    @contextmanager
    def _use_reference(self):
        live = self._snapshot_lora()
        try:
            self._copy_lora(self.reference)
            yield
        finally:
            self._copy_lora(live)

    def _commit(self, step):
        scaler = self.accelerator.scaler
        state = dict(fingerprint=self.fingerprint, run_id=self.run_id, step=step,
                     live=self._snapshot_lora(), reference=self.reference,
                     optimizer=self.optimizer.state_dict(), scheduler=self.lr_scheduler.state_dict(),
                     scaler=scaler.state_dict() if scaler is not None else None,
                     budget=self.budget, drift_ema=self.drift_ema, multiplier=self.multiplier)
        write_atomic(self.state_path, lambda file: torch.save(state, file))
        self.committed_step = self.last_save_step = step

    @torch.no_grad()
    def _generate_round(self, round_dir, step):
        cfg = self.sifr
        if self.sd.transformer.device.type == "cpu":
            self.sd.transformer.to(self.device_torch)
        pipeline = self.sd.get_generation_pipeline()
        manifest = {"round_id": uuid.uuid4().hex, "run_id": self.run_id, "step": step, "samples": []}
        latents = []
        vae_device = self.sd.vae.device
        reference = self._use_reference() if cfg.rollout_policy == "ema" else nullcontext()
        try:
            with reference:
                for group, prompt in enumerate(self.prompts):
                    for _ in range(cfg.samples_per_prompt):
                        self.maybe_stop()
                        index = len(latents)
                        sample_id = f"sample-{index:04d}"
                        seed = (cfg.seed + step * len(self.prompts) * cfg.samples_per_prompt + index) % (2**63)
                        self.update_status("running", f"Generating SIFR sample {index + 1}/{len(self.prompts) * cfg.samples_per_prompt}")
                        generator = torch.Generator().manual_seed(seed)
                        images, clean = pipeline(
                            self.prompt_embeds[group], self.negative_embeds,
                            height=cfg.height, width=cfg.width, num_inference_steps=cfg.sample_steps,
                            guidance_scale=cfg.guidance_scale, generator=generator,
                            shift=self.sd.sample_shift, uncond_skip_layers=self.sd.uncond_skip_layers,
                            return_latents=True,
                        )
                        images[0].save(round_dir / f"{sample_id}.png")
                        latents.append(clean.cpu())
                        manifest["samples"].append(dict(id=sample_id, prompt=prompt, group=group, seed=str(seed)))
        finally:
            self.sd.vae.to(vae_device)
            flush()
        write_atomic(round_dir / "rollouts.pt", lambda file: torch.save(torch.cat(latents), file))
        template = {"round_id": manifest["round_id"], "scores": {s["id"]: None for s in manifest["samples"]}}
        write_json(round_dir / "ratings.template.json", template)
        write_json(round_dir / "manifest.json", manifest)
        return manifest

    def _wait_for_scores(self, scores_path, manifest):
        print_acc(f"\nWaiting for SIFR ratings in the job's Ratings tab or at {scores_path}")
        self.update_status("running", f"Waiting for {len(manifest['samples'])} SIFR ratings")
        previous_error = None
        while True:
            self.maybe_stop()
            try:
                return read_scores(scores_path, manifest)
            except FileNotFoundError:
                pass
            except ValueError as error:
                if str(error) != previous_error:
                    previous_error = str(error)
                    print_acc(f"SIFR ratings: {error}")
            time.sleep(2)

    def _round_generator(self, offset):
        seed = (self.sifr.seed + self.committed_step * 1_000_003 + offset) % (2**63)
        return torch.Generator().manual_seed(seed)

    def _batch(self, indices, generator):
        noisy, t, target = flow_batch(self.rollout_latents[indices], generator, self.device_torch)
        embeds = AdvancedPromptEmbeds.concat_prompt_embeds([
            self.prompt_embeds[index // self.sifr.samples_per_prompt] for index in indices
        ])
        return noisy, t, target, embeds

    def _predict(self, noisy, timestep, embeds):
        with self.accelerator.autocast():
            return self.sd.get_noise_prediction(noisy, timestep, embeds)

    @torch.no_grad()
    def _measure_round(self, reference_predictions=None):
        # Both passes use the same held-out noise and timesteps across all rollouts.
        generator = self._round_generator(2**40)
        predictions, errors = [], []
        indices = torch.arange(len(self.rollout_latents))
        for chunk in indices.split(self.train_config.batch_size):
            self.maybe_stop()
            noisy, t, target, embeds = self._batch(chunk, generator)
            prediction = self._predict(noisy, t, embeds).cpu()
            if reference_predictions is None:
                predictions.append(prediction)
                target = target.cpu()
            else:
                target = reference_predictions[chunk]
            errors.append(velocity_mse(prediction, target))
        error = torch.cat(errors).mean().item()
        if not math.isfinite(error):
            raise FloatingPointError("Nonfinite SIFR drift or budget estimate")
        return (torch.cat(predictions) if predictions else None), error

    def _initialize_sifr(self):
        self.reference = self._snapshot_lora()
        self._commit(self.step_num)

    def _begin_round(self):
        if self.committed_step is None:
            self._initialize_sifr()
        self.sd.text_encoder_to("cpu")
        if self.model_config.low_vram:
            self.sd.vae.to("cpu")
        flush()
        round_dir = self.feedback_root / f"round-{self.step_num:09d}"
        round_dir.mkdir(exist_ok=True)
        manifest_path = round_dir / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if (manifest["run_id"], manifest["step"]) != (self.run_id, self.step_num):
                raise ValueError("SIFR rollouts belong to another run. Use a new job name")
        else:
            manifest = self._generate_round(round_dir, self.step_num)
        write_json(self.active_path, manifest)
        scores = self._wait_for_scores(round_dir / "scores.json", manifest)
        self.update_status("running", "Training on SIFR ratings")
        self.rollout_latents = torch.load(round_dir / "rollouts.pt", map_location="cpu", weights_only=True)
        self.advantages = signed_advantages(scores, self.sifr.samples_per_prompt)
        with self._use_reference():
            self.reference_predictions, baseline_loss = self._measure_round()
        if self.budget is None:
            self.budget = max(self.sifr.budget_ratio * baseline_loss, 1e-12)
        self.round_dir = round_dir

    def _finish_round(self, step):
        _, drift = self._measure_round(self.reference_predictions)
        cfg = self.sifr
        self.drift_ema = drift if self.drift_ema is None else cfg.drift_ema_decay * self.drift_ema + (1 - cfg.drift_ema_decay) * drift
        self.multiplier = update_multiplier(self.multiplier, self.drift_ema, self.budget,
                                            cfg.lambda_rate, cfg.lambda_min, cfg.lambda_max)
        for name, parameter in self.lora_parameters.items():
            self.reference[name].lerp_(parameter.detach().cpu(), 1 - cfg.reference_decay)
        print_acc(f"SIFR step {step}: drift={drift:.4g}, budget={self.budget:.4g}, next lambda={self.multiplier:.4g}")
        self._commit(step)
        self.active_path.unlink(missing_ok=True)
        (self.round_dir / "rollouts.pt").unlink(missing_ok=True)
        self.round_dir = self.rollout_latents = self.reference_predictions = None

    def hook_train_loop(self, batch):
        warming_up = self.step_num < self.sifr.warmup_steps
        try:
            if warming_up:
                loss = super().hook_train_loop(batch)
                if self.accelerator.optimizer_step_was_skipped:
                    raise FloatingPointError("An optimizer update was skipped during SIFR supervised warmup")
                return loss
            self.network.eval()
            self.network.is_active = True
            self.network.multiplier = 1.0
            self.sd.transformer.eval()
            if self.round_dir is None:
                self._begin_round()
            return self._optimize_step()
        except RuntimeError as error:
            if isinstance(error, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(error):
                # Neither phase can count a skipped batch as a completed update.
                phase = "SIFR supervised warmup" if warming_up else "SIFR"
                raise RuntimeError(f"GPU memory exhausted during {phase}. Restart to resume from the last checkpoint.") from error
            raise

    def _optimize_step(self):
        offset = self.step_num - self.committed_step
        count = len(self.rollout_latents)
        batch_size = self.train_config.batch_size
        cycle, batch_index = divmod(offset, math.ceil(count / batch_size))
        order = torch.randperm(count, generator=self._round_generator(cycle))
        indices = order[batch_index * batch_size:(batch_index + 1) * batch_size]
        noisy, t, target, embeds = self._batch(indices, self._round_generator(2**32 + offset))
        with self._use_reference(), torch.no_grad():
            reference = self._predict(noisy, t, embeds)
        prediction = self._predict(noisy, t, embeds)
        signed = (self.advantages[indices].to(prediction.device) * velocity_mse(prediction, target)).mean()
        drift = velocity_mse(prediction, reference).mean()
        loss = signed + self.multiplier * drift
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite SIFR loss")
        self.optimizer.zero_grad(set_to_none=True)
        self.accelerator.backward(loss)
        sync_grad_transfers()
        self.accelerator.clip_grad_norm_(self.lora_parameters.values(), self.train_config.max_grad_norm)
        self.optimizer.step()
        if self.accelerator.optimizer_step_was_skipped:
            raise FloatingPointError("An optimizer update was skipped during SIFR")
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        metrics = {"loss": loss.item(), "sifr_signed": signed.item(), "sifr_drift": drift.item(), "sifr_lambda": self.multiplier}
        if offset + 1 == self.sifr.rollout_every or self.step_num + 1 == self.train_config.steps:
            self._finish_round(self.step_num + 1)
        return metrics

    def save(self, step=None):
        if self.committed_step is not None and self.step_num != self.committed_step:
            return  # Export only from a completed, resumable round.
        super().save(step)
        self.last_export_step = self.step_num

    def maybe_save(self):
        if self.committed_step is None or self.step_num == self.committed_step:
            super().maybe_save()

    def end_step_hook(self):
        if self.sifr.warmup_steps and self.step_num == self.sifr.warmup_steps and self.committed_step is None:
            self._initialize_sifr()
            self.save(self.step_num)
            print_acc(f"Supervised warmup complete at step {self.step_num}. SIFR review begins next.")

        super().end_step_hook()
        every = self.checkpoint_every
        if (every and self.step_num < self.train_config.steps and self.step_num - self.last_export_step >= every
                and (self.committed_step is None or self.step_num == self.committed_step)):
            self.save(self.step_num)
