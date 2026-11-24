# Human-rated SIFR for MOSS Image

This trainer applies [Stable Signed Flow Reinforcement](https://arxiv.org/abs/2611.09175) to a MOSS Image text-to-image LoRA. The transformer and text encoder stay frozen. Rate generated images against their prompt and your preference. The trainer normalizes scores within each prompt group, uses the signed advantages for flow regression, and adapts a velocity-drift penalty against an EMA LoRA reference.

## Run a job

1. Open [the example config](../../config/examples/train_moss_image_sifr.yaml) and replace `sifr.prompts` with prompts you want to optimize for. Use `network.pretrained_lora_path` to initialize from an existing MOSS Image LoRA, or let the toolkit create one.
2. In AI Toolkit, choose **New Job → Show Advanced → Import Config**, import the YAML, and start the job. From the command line, run `python run.py config/examples/train_moss_image_sifr.yaml` with the toolkit's Python environment.
3. Open the running job's **Ratings** tab, score every image from 1 to 5, and click **Submit ratings**. Training resumes automatically. Equal scores within a prompt group give that group zero signed reward signal.

For CLI jobs, the log prints the round folder. View the PNGs in that folder, fill in `ratings.template.json` with an integer score for every image, and save the completed copy as `scores.json` alongside the images. Keep the round ID intact. The trainer checks for the file every two seconds.

## Timing and settings

`sifr.rollout_every` is the number of optimizer steps between reviews. With no warmup the first review is at step 0. To start with supervised training, add a dataset and set `sifr.warmup_steps` to the number of supervised steps before switching to reviews. `train.steps` is the total step count including warmup. With the example settings, reviews occur at steps 0, 8, and 16, followed by the final LoRA save at step 24.

The [Kromoss candid-photo recipe](../../config/examples/train_kromoss_candid_sifr.yaml) uses 5,000 supervised updates on captioned images, followed by 80 SIFR updates. Replace `datasets[0].folder_path` with your training folder and keep each caption beside its image with the same filename stem. Import the config into AI Toolkit or pass it to `python run.py`. The trainer saves the warmup LoRA at completed step 5,000, releases its dataset loaders, and pauses for its first SIFR ratings. Later reviews occur every 16 updates. Restarting the job from its SIFR checkpoint resumes from the last completed round and skips the supervised dataset caches.

`sifr.rollout_policy: ema` samples with the paper's round-based EMA LoRA. `live` samples with the exact current LoRA and still uses the EMA LoRA for the drift penalty. The rollout uses the MOSS Euler sampler and stores the final generated latents. The regression uses fresh Gaussian noise, uniform training timesteps, and conditional velocity. The first round calibrates the drift budget from its own samples. `reference_decay`, `drift_ema_decay`, `budget_ratio`, and the `lambda_*` settings are configurable in `sifr`.

Run one training process using the MOSS Image text-to-image architecture and a transformer LoRA. `train.batch_size` controls how many reviewed latents enter each optimizer step. `train.disable_sampling` controls the toolkit's ordinary previews independently of SIFR reviews.

The job writes ordinary LoRA checkpoints plus `sifr/checkpoint.pt`, which stores the training weights, EMA reference, optimizer, and drift controller at round boundaries. Restart the same job to reuse pending images and submitted ratings. An interruption during updates replays that block from its preceding checkpoint. Keep the `sifr` folder with the job to preserve resume state. Use a new job name for changed SIFR settings.
