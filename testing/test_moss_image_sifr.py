import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from extensions_built_in.moss_image_sifr.feedback import read_scores, write_json
from extensions_built_in.moss_image_sifr.objective import (
    flow_batch, signed_advantages, update_multiplier, velocity_mse,
)


def test_signed_flow_loss_has_opposite_gradients_for_high_and_low_ratings():
    advantages = signed_advantages([1, 5, 3, 3], samples_per_prompt=2)
    torch.testing.assert_close(advantages, torch.tensor([-1.0, 1.0, 0.0, 0.0]))
    prediction = torch.zeros(4, 1, 1, 1, requires_grad=True)
    loss = (advantages * velocity_mse(prediction, torch.ones_like(prediction))).mean()
    loss.backward()
    torch.testing.assert_close(prediction.grad.flatten(), torch.tensor([0.5, -0.5, 0.0, 0.0]))


def test_forward_flow_follows_the_moss_noise_to_image_convention():
    clean = torch.ones(4, 2, 2, 2)
    generator = torch.Generator().manual_seed(2)
    noisy, timestep, target = flow_batch(clean, generator, clean.device)
    reconstructed = noisy - (timestep[:, None, None, None] / 1000) * target
    torch.testing.assert_close(reconstructed, clean)
    assert (0 <= timestep).all() and (timestep < 1000).all()


def test_multiplier_reacts_to_drift_and_respects_its_bounds():
    assert update_multiplier(1, 1, 1, 0.1, 0.5, 2) == 1
    assert update_multiplier(1, 2, 1, 0.1, 0.5, 2) > 1
    assert update_multiplier(1, 0, 1, 0.1, 0.5, 2) < 1
    assert update_multiplier(1, 100, 1, 10, 0.5, 2) == 2
    assert update_multiplier(1, 0, 1, 10, 0.5, 2) == 0.5


def test_ratings_are_complete_and_bound_to_the_expected_round(tmp_path):
    path = tmp_path / 'scores.json'
    manifest = {'round_id': 'current', 'samples': [{'id': 'a'}, {'id': 'b'}]}
    write_json(path, {'round_id': 'current', 'scores': {'b': 4, 'a': 2}})
    assert read_scores(path, manifest) == [2, 4]

    for invalid in [
        {'round_id': 'old', 'scores': {'a': 2, 'b': 4}},
        {'round_id': 'current', 'scores': {'a': 2}},
        {'round_id': 'current', 'scores': {'a': True, 'b': 4}},
        {'round_id': 'current', 'scores': {'a': 2, 'b': 6}},
    ]:
        write_json(path, invalid)
        with pytest.raises(ValueError):
            read_scores(path, manifest)

    path.write_text('{', encoding='utf-8')
    with pytest.raises(json.JSONDecodeError):
        read_scores(path, manifest)


@pytest.fixture
def isolated_sifr_trainer(monkeypatch):
    # Run the actual SIFR phase hooks on CPU without loading the model registry.
    class SupervisedTrainer:
        def hook_train_loop(self, batch):
            assert batch == ["dataset batch"]
            self.optimizer.zero_grad()
            loss = self.network.weight.square().sum()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            return {"loss": loss.item()}

        def end_step_hook(self):
            pass

        def save(self, step):
            self.exports.append(step)

    parent = ModuleType("extensions_built_in.sd_trainer.DiffusionTrainer")
    parent.DiffusionTrainer = SupervisedTrainer
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    memory_management = ModuleType("toolkit.memory_management")
    memory_management.sync_grad_transfers = lambda: None
    monkeypatch.setitem(sys.modules, memory_management.__name__, memory_management)

    name = "extensions_built_in.moss_image_sifr._test_trainer"
    path = Path(__file__).resolve().parents[1] / "extensions_built_in/moss_image_sifr/trainer.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_warmup_boundary_exports_the_supervised_weights_before_ratings(isolated_sifr_trainer, tmp_path):
    module = isolated_sifr_trainer
    trainer = module.MossImageSIFRTrainer.__new__(module.MossImageSIFRTrainer)
    trainer.sifr = module.SIFRConfig(prompts=["a candid street photo"], warmup_steps=5000)
    trainer.step_num = 4999
    trainer.train_config = SimpleNamespace(steps=5080)
    trainer.state_path = tmp_path / "checkpoint.pt"
    trainer.accelerator = SimpleNamespace(scaler=None, optimizer_step_was_skipped=False)
    trainer.budget = trainer.drift_ema = trainer.committed_step = None
    trainer.multiplier = 0.01
    trainer.fingerprint, trainer.run_id = "configuration", "run"
    trainer.network = torch.nn.Linear(1, 1, bias=False)
    trainer.network.weight.data.fill_(1.0)
    trainer.lora_parameters = dict(trainer.network.named_parameters())
    trainer.sd = SimpleNamespace(transformer=torch.nn.Identity())
    trainer.optimizer = torch.optim.AdamW(trainer.network.parameters(), lr=0.01)
    trainer.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(trainer.optimizer, factor=1.0)
    trainer.exports = []
    trainer.checkpoint_every, trainer.last_export_step = 250, 4750
    trainer.round_dir = None

    assert trainer.needs_training_dataset(4999)
    assert trainer.hook_train_loop(["dataset batch"])["loss"] == 1.0
    assert trainer.network.weight.item() < 1.0 and not trainer.state_path.exists()

    trainer.step_num = 5000
    trainer.end_step_hook()
    state = torch.load(trainer.state_path, weights_only=True)
    assert trainer.exports == [5000] and trainer.committed_step == state["step"] == 5000
    torch.testing.assert_close(state["live"]["weight"], trainer.network.weight)
    torch.testing.assert_close(state["reference"]["weight"], state["live"]["weight"])
    assert not trainer.needs_training_dataset(5000)
    assert not trainer.needs_training_dataset(4750)  # An older export cannot restart warmup.

    trainer._begin_round = lambda: setattr(trainer, "round_dir", "first review")
    trainer._optimize_step = lambda: {"loss": 0.5}
    assert trainer.hook_train_loop([None]) == {"loss": 0.5}
    assert trainer.round_dir == "first review"
