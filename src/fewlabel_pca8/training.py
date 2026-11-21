import hashlib
from pathlib import Path

import torch

from fewlabel.runtime import write_json
from fewlabel_v21.training import with_model_factory
from .models import (build_model, EXPECTED_PARAMETERS, EXPECTED_TRAINABLE_PARAMETERS,
                     FROZEN_PARAMETER_NAMES, FROZEN_BUFFER_NAMES, IDENTITY_FIELDS)


def projection_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_audited_trainer(factory=build_model):

    state = {"gradient_checks": 0}

    def captured_factory(name, cfg):
        if "student" in state:
            raise RuntimeError("An audited trainer instance may construct only one student")
        model = factory(name, cfg)
        parameters = dict(model.named_parameters())
        if tuple(key for key, parameter in parameters.items()
                 if not parameter.requires_grad) != FROZEN_PARAMETER_NAMES:
            raise RuntimeError("Unexpected trainability contract")
        state["initial_projection"] = parameters[FROZEN_PARAMETER_NAMES[0]].detach().cpu().clone()
        state["initial_bias"] = dict(model.named_buffers())[FROZEN_BUFFER_NAMES[0]].detach().cpu().clone()
        state['identity'] = {key: getattr(model, key) for key in IDENTITY_FIELDS}
        state["student"] = model
        return model

    trainer = with_model_factory(captured_factory)
    original_check = trainer.__globals__["_finite_gradients"]

    def checked_gradients(model):
        if model is not state.get("student"):
            raise RuntimeError("Frozen-weight audit saw a different student")
        parameters = dict(model.named_parameters())
        if tuple(key for key, parameter in parameters.items()
                 if not parameter.requires_grad) != FROZEN_PARAMETER_NAMES:
            raise RuntimeError("Trainability changed during the fit")
        weight = parameters[FROZEN_PARAMETER_NAMES[0]]
        if weight.grad is not None:
            raise RuntimeError("Frozen projection acquired a gradient")
        if "device_reference" not in state:
            state["device_reference"] = state["initial_projection"].to(weight.device)
        if not torch.equal(weight.detach(), state["device_reference"]):
            raise RuntimeError("Frozen projection changed before an optimizer step")
        bias = dict(model.named_buffers())[FROZEN_BUFFER_NAMES[0]]
        if 'device_bias_reference' not in state:
            state['device_bias_reference'] = state['initial_bias'].to(bias.device)
        if bias.requires_grad or not torch.equal(bias, state['device_bias_reference']):
            raise RuntimeError('Frozen PCA bias changed before an optimizer step')
        if any(parameter.grad is None for parameter in parameters.values() if parameter.requires_grad):
            raise RuntimeError("A trainable parameter has no gradient")
        norm = original_check(model)
        state["gradient_checks"] += 1
        return norm

    trainer.__globals__["_finite_gradients"] = checked_gradients
    return trainer, state


def finish_audit(state, result, out_dir):

    parameters = dict(state["student"].named_parameters())
    frozen = parameters[FROZEN_PARAMETER_NAMES[0]]
    if frozen.requires_grad or frozen.grad is not None:
        raise RuntimeError("Frozen projection trainability or gradient changed at completion")
    total = sum(parameter.numel() for parameter in parameters.values())
    trainable = sum(parameter.numel() for parameter in parameters.values() if parameter.requires_grad)
    if total != EXPECTED_PARAMETERS or trainable != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("Parameter counts changed at completion")
    expected_steps = result["epochs_completed"] * result["steps_per_epoch"]
    if state["gradient_checks"] != expected_steps:
        raise RuntimeError("Not every optimizer step has frozen-weight evidence")
    checkpoint = torch.load(Path(out_dir) / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("model") != result["model"] or checkpoint.get("seed") != result["seed"]:
        raise RuntimeError("Selected checkpoint identity mismatch")
    initial = state["initial_projection"]
    final = frozen.detach().cpu()
    best = checkpoint["state_dict"][FROZEN_PARAMETER_NAMES[0]]
    if not torch.equal(initial, final) or not torch.equal(initial, best):
        raise RuntimeError("Frozen projection differs from initialization in best or final state")
    initial_bias = state['initial_bias']
    final_bias = dict(state['student'].named_buffers())[FROZEN_BUFFER_NAMES[0]].detach().cpu()
    best_bias = checkpoint['state_dict'][FROZEN_BUFFER_NAMES[0]]
    if not torch.equal(initial_bias, final_bias) or not torch.equal(initial_bias, best_bias):
        raise RuntimeError('Frozen PCA bias differs in best or final state')
    return {
        'status': 'PASS',
        "frozen_parameter_names": list(FROZEN_PARAMETER_NAMES),
        'frozen_buffer_names': list(FROZEN_BUFFER_NAMES),
        "total_parameters": total, "trainable_parameters": trainable,
        "initial_projection_sha256": projection_digest(initial),
        "final_projection_sha256": projection_digest(final),
        "best_projection_sha256": projection_digest(best),
        'initial_bias_sha256': projection_digest(initial_bias),
        'final_bias_sha256': projection_digest(final_bias),
        'best_bias_sha256': projection_digest(best_bias),
        "gradient_checks": state["gradient_checks"],
        "frozen_gradient_absent_every_step": True,
        "best_and_final_equal_initial": True,
        "scope": "SENTETIK_TEST" if result.get("scope") == "SENTETIK_TEST" else "FROZEN_WEIGHT_INTEGRITY",
    }


def defer_result_write(trainer, out_dir, state=None):

    original_write = trainer.__globals__["write_json"]
    result_path = (Path(out_dir) / "result.json").resolve()

    def checked_write(path, value):
        if Path(path).resolve() == result_path:
            return None
        if Path(path).name == 'effective_config.json' and state is not None:
            value = dict(value, **state['identity'])
        return original_write(path, value)

    trainer.__globals__["write_json"] = checked_write


def train_job(root, cfg, model_name, seed, prepared, out_dir, status, max_epochs=None,
              profile=False, progress_callback=None):
    trainer, state = make_audited_trainer()
    defer_result_write(trainer, out_dir, state)
    try:
        result = trainer(root, cfg, model_name, seed, prepared, out_dir, status,
                         max_epochs=max_epochs, profile=profile, progress_callback=progress_callback)
        result['pca_projection_audit'] = finish_audit(state, result, out_dir)
        result.update(state['identity'])
        write_json(Path(out_dir) / "result.json", result)
        return result
    finally:
        state.clear()
