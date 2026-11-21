from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from fewlabel.runtime import clean, derived_seed, now, sha
from fewlabel.training import (dice_gradient_coefficients, dice_gradient_surrogate,
                              _finite_gradients, validate)
from fewlabel.objectives import dice_from_statistics, ema_update
from colon_spectr_recovery.objectives import masked_supervised
from colon_nnunet_recovery.runtime import write_json, replace_with_backoff
from .data import UniformPatchPlan
from .models import build_model, PROJECTION, state_digest


def atomic_torch(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as handle:
        torch.save(payload, handle); handle.flush(); os.fsync(handle.fileno())
    replace_with_backoff(temporary, path)


def _atomic_npz(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
    replace_with_backoff(temporary, path)


class SelectionTracker:

    def __init__(self, patience=15):
        self.patience = int(patience)
        self.best_score, self.best_ce = -float('inf'), float('inf')
        self.best_epoch, self.stale, self.virtual_stop_epoch = 0, 0, None
        self.virtual_best_epoch = None

    def observe(self, epoch, score, ce):
        improved = score > self.best_score or (score == self.best_score and ce < self.best_ce)
        if improved:
            self.best_score, self.best_ce, self.best_epoch, self.stale = score, ce, epoch, 0
        else:
            self.stale += 1
        triggered = self.virtual_stop_epoch is None and self.stale >= self.patience
        if triggered:
            self.virtual_stop_epoch, self.virtual_best_epoch = epoch, self.best_epoch
        return improved, triggered


def effective_batch_backward(student, batches, device):


    totals = {key: torch.zeros(2, device=device) for key in
              ('dice_intersection', 'dice_prediction_sum', 'dice_target_sum')}
    total_labels, ce_value = 0, 0.
    with torch.no_grad():
        for inputs, labels in batches:
            stats = masked_supervised(student(inputs.to(device))['logits'], labels.to(device))
            total_labels += int(stats['labeled_count'])
            ce_value += float(stats['ce_sum'])
            for key in totals:
                totals[key] += stats[key]
    coefficients = dice_gradient_coefficients(**{
        'intersection': totals['dice_intersection'],
        'prediction_sum': totals['dice_prediction_sum'],
        'target_sum': totals['dice_target_sum']})
    for inputs, labels in batches:
        stats = masked_supervised(student(inputs.to(device))['logits'], labels.to(device))
        loss = stats['ce_sum']/max(1, total_labels) + dice_gradient_surrogate(stats, coefficients)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite mask-study objective')
        loss.backward()
    dice = float(dice_from_statistics(totals['dice_intersection'], totals['dice_prediction_sum'],
                                      totals['dice_target_sum']))
    return {'ce': ce_value/max(1, total_labels), 'dice': dice,
            'loss': ce_value/max(1, total_labels)+dice, 'labeled_presentations': total_labels}


def configure_strict(init_seed):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before process CUDA initialization')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(int(init_seed)); torch.cuda.manual_seed_all(int(init_seed))
    np.random.seed(int(init_seed)); random.seed(int(init_seed))


def train_job(root, cfg, model_name, mask_seed, init_seed, prepared, out_dir, status,
              initial_path, progress_callback=None):

    out_dir, root = Path(out_dir), Path(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir/name).exists() for name in ('result.json', 'epochs.jsonl', 'last.pt')):
        raise RuntimeError('Refuse automatic retry or overwrite of an existing fit')
    spec = cfg['maskstudy']
    epochs, steps = int(spec.get('epochs', 100)), int(cfg['pilot']['steps_per_epoch'])
    effective, physical = int(cfg['train']['effective_batch_size']), int(cfg['train']['physical_batch_size'])
    if (epochs, steps, effective, physical) != (100, 128, 32, 2):
        raise ValueError('Full study requires 100 epochs, 128 steps, effective32/physical2')
    if cfg['ssl']['lambda_u_max'] != 0 or cfg['ssl'].get('lambda_proto_max', 0) != 0:


        raise ValueError('Mask study is supervised-only: both auxiliary weights must be zero')
    if not spec.get('strict_deterministic', True):
        raise ValueError('Mask study requires strict deterministic numerical execution')
    if (prepared.train_mask_seed != int(mask_seed)
            or prepared.validation_mask_seed != int(spec.get('common_val_seed', 938))):
        raise RuntimeError('Composed mask identity mismatch')
    if set(prepared.train_ids) & set(prepared.val_ids):
        raise RuntimeError('TRAIN/VAL overlap')
    configure_strict(init_seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Production mask-study training requires CUDA; no fallback')
    device = torch.device('cuda')
    student = build_model(model_name, cfg, initial_path, init_seed).to(device)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(cfg['train']['lr']),
                                  weight_decay=float(cfg['train']['weight_decay']))
    frozen_reference = dict(student.named_parameters())[PROJECTION].detach().clone()
    start = time.monotonic()
    initial_hash = state_digest(student.state_dict())
    metadata = {'model': model_name, 'mask_seed': int(mask_seed), 'init_seed': int(init_seed),
        'data_seed': int(spec['data_seed']), 'common_val_seed': prepared.validation_mask_seed,
        'initial_weights_sha256': initial_hash, 'initial_file_sha256': sha(initial_path),
        'common_plan_sha256': sha(spec['plan_path']), 'mask_sha256': dict(prepared.mask_sha256),
        'config': cfg, 'epochs': epochs, 'steps_per_epoch': steps, 'effective_batch': effective,
        'physical_batch': physical, 'test_access': False, 'precision': 'float32',
        'numerics': 'STRICT_DETERMINISTIC_CE_NONE_SUM_BILINEAR_MATRIX',
        'teacher_forward_and_ema_preserved': True,
        'inactive_consistency_prototype_diagnostics_omitted': True,
        'restart_policy': 'NO_AUTOMATIC_RETRY', 'primary_endpoint': 'fixed_end_12800'}
    write_json(out_dir/'effective_config.json', metadata)
    seen = {image: np.zeros(np.prod(prepared.image_shape(image)), dtype=np.bool_)
            for image in prepared.train_ids}
    tracker = SelectionTracker(15)
    history, endpoints, best_payload, best_validation, best_predictions = [], {}, None, None, None
    total_presentations = empty_effective = empty_micro = 0
    class_presentations = np.zeros(2, dtype=np.int64)
    torch.cuda.reset_peak_memory_stats()

    def progress():
        if progress_callback:
            progress_callback(time.monotonic()-start)

    def save_endpoint(name, payload, validation, predictions, selection_epoch, stop_epoch=None):
        atomic_torch(out_dir/f'{name}.pt', payload)
        write_json(out_dir/f'{name}_validation.json', validation)
        paths = [out_dir/f'{name}.pt', out_dir/f'{name}_validation.json']
        for image, arrays in predictions.items():
            path = out_dir/f'{name}_sparsevalpred_{image}.npz'
            _atomic_npz(path, arrays); paths.append(path)
        endpoints[name] = {'selected_epoch': int(selection_epoch), 'virtual_stop_epoch': stop_epoch,
            'optimizer_steps_at_checkpoint': int(selection_epoch)*steps,
            'image_mean_macro_f1': validation['summary']['image_mean_macro_f1'],
            'sparse_ce': validation['sparse_ce'],
            'artifacts': {p.name: sha(p) for p in paths}}

    for epoch in range(1, epochs+1):
        epoch_start = time.monotonic(); student.train()
        plan = UniformPatchPlan(prepared, cfg, int(spec['data_seed']), epoch, effective*steps)
        workers = int(cfg['train'].get('num_workers', 4))
        options = {'batch_size': physical, 'shuffle': False, 'num_workers': workers,
            'pin_memory': True, 'drop_last': False,
            'generator': torch.Generator().manual_seed(derived_seed(spec['data_seed'], 'maskstudy_loader', epoch))}
        if workers:
            options.update(prefetch_factor=int(cfg['train'].get('prefetch_factor', 4)), persistent_workers=False)
        loader = iter(DataLoader(plan, **options))
        losses, epoch_labels, epoch_empty_batches = [], 0, 0
        input_hashes = {}
        for step in range(1, steps+1):
            cached = []
            for micro in range(effective//physical):
                batch = next(loader)
                labels, inputs = batch['labels'], batch['strong']
                labeled = labels.ge(0)
                empty_micro += int(not bool(labeled.any()))
                for index, image_index in enumerate(batch['image_index'].tolist()):
                    valid = labeled[index]
                    coords = batch['coordinates'][index][valid].numpy()
                    seen[prepared.train_ids[image_index]][coords] = True
                if bool(labeled.any()):
                    class_presentations += torch.bincount(labels[labeled], minlength=2).numpy()
                if micro == 0 and step in (1, steps):
                    input_hashes[str(step)] = hashlib.sha256(inputs.contiguous().numpy().tobytes()).hexdigest()


                with torch.no_grad():
                    teacher_output = teacher(batch['weak'].to(device, non_blocking=True))
                    if not bool(torch.isfinite(teacher_output['logits']).all()):
                        raise FloatingPointError('Nonfinite teacher output')
                cached.append((inputs, labels))
            optimizer.zero_grad(set_to_none=True)
            values = effective_batch_backward(student, cached, device)
            gradient_norm = _finite_gradients(student)
            if any(p.grad is None for p in student.parameters() if p.requires_grad):
                raise RuntimeError('Trainable parameter lacks gradient')
            if model_name == 'local_frozen_projection':
                weight = dict(student.named_parameters())[PROJECTION]
                if weight.grad is not None or not torch.equal(weight, frozen_reference):
                    raise RuntimeError('Fixed projection changed or received gradients')
            optimizer.step()
            ema_update(teacher, student, float(cfg['ssl']['ema_teacher']))
            n = values['labeled_presentations']
            epoch_labels += n; total_presentations += n
            epoch_empty_batches += int(n == 0); empty_effective += int(n == 0)
            losses.append(values['loss'])
            if status is not None:
                status.update(status='RUNNING', stage='training', model=model_name,
                    mask_seed=int(mask_seed), init_seed=int(init_seed), epoch=epoch,
                    max_epochs=epochs, step=step, steps_per_epoch=steps,
                    optimizer_steps_completed=(epoch-1)*steps+step,
                    last_metrics={**values, 'gradient_norm': gradient_norm},
                    best=tracker.best_score if tracker.best_epoch else None,
                    best_epoch=tracker.best_epoch, early_stopping_bad_epochs=tracker.stale)
            progress()
        del loader, cached
        torch.cuda.synchronize(); train_seconds = time.monotonic()-epoch_start
        validation_start = time.monotonic()
        if status is not None:
            status.update(stage='validation', epoch=epoch)
        validation, predictions = validate(student, prepared, cfg, physical, device, progress)
        validation_seconds = time.monotonic()-validation_start
        score, ce = validation['summary']['image_mean_macro_f1'], validation['sparse_ce']
        improved, triggered = tracker.observe(epoch, score, ce)
        payload = {'state_dict': {k: v.detach().cpu().clone() for k,v in student.state_dict().items()},
            'model': model_name, 'mask_seed': int(mask_seed), 'init_seed': int(init_seed),
            'epoch': epoch, 'optimizer_steps': epoch*steps, 'prediction_model': 'student',
            'initial_weights_sha256': initial_hash}
        if improved:
            best_payload, best_validation, best_predictions = payload, validation, predictions
            save_endpoint('best_full_budget', best_payload, best_validation, best_predictions, epoch)
        if triggered:
            save_endpoint('virtual_es15', best_payload, best_validation, best_predictions,
                          tracker.virtual_best_epoch, tracker.virtual_stop_epoch)


        np_state = np.random.get_state()
        full_state = {**payload, 'optimizer_state': optimizer.state_dict(),
            'teacher_state': teacher.state_dict(), 'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_states': torch.cuda.get_rng_state_all(), 'python_rng_state': random.getstate(),
            'numpy_rng_state': (np_state[0], np_state[1].tolist(), *np_state[2:]),
            'resume_authorized': False}
        atomic_torch(out_dir/'last.pt', full_state)
        epoch_seconds = time.monotonic()-epoch_start
        mean_epoch = (time.monotonic()-start)/epoch
        record = {'timestamp': now(), 'epoch': epoch, 'optimizer_steps': steps,
            'optimizer_steps_completed': epoch*steps, 'sampled_patches': effective*steps,
            'loss': float(np.mean(losses)), 'validation': validation, 'val_image_macro_f1': score,
            'best_epoch': tracker.best_epoch, 'best_val_image_macro_f1': tracker.best_score,
            'early_stopping_bad_epochs': tracker.stale,
            'virtual_stop_epoch': tracker.virtual_stop_epoch, 'early_stopping_enabled': False,
            'revealed_label_presentations': epoch_labels,
            'unique_revealed_labels_seen_cumulative': {im: int(v.sum()) for im,v in seen.items()},
            'empty_effective_batches': epoch_empty_batches,
            'coordinate_sha256': plan.coordinate_sha256, 'strong_input_probe_sha256': input_hashes,
            'train_s': train_seconds, 'validation_s': validation_seconds, 'epoch_s': epoch_seconds,
            'mean_epoch_s': mean_epoch, 'remaining_fit_estimate_s': (epochs-epoch)*mean_epoch,
            'elapsed_s': time.monotonic()-start}
        history.append(record)
        with (out_dir/'epochs.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(clean(record), ensure_ascii=False, allow_nan=False)+'\n')
            handle.flush(); os.fsync(handle.fileno())
        if status is not None:
            status.update(stage='epoch_complete', epoch=epoch, best=tracker.best_score,
                best_epoch=tracker.best_epoch, early_stopping_bad_epochs=tracker.stale,
                virtual_es15_trigger_epoch=tracker.virtual_stop_epoch,
                mean_epoch_s=mean_epoch, remaining_fit_estimate_s=(epochs-epoch)*mean_epoch,
                last_metrics={'val_f1': score, 'loss': record['loss'], 'epoch_s': epoch_seconds},
                history=[{'epoch':r['epoch'],'val_f1':r['val_image_macro_f1'],'loss':r['loss']} for r in history])
            status.event('epoch_complete', model=model_name, mask_seed=int(mask_seed),
                         init_seed=int(init_seed), epoch=epoch, val_image_macro_f1=score)
        print(f'{epoch:03d}/100 | {model_name} | mask={mask_seed} init={init_seed} | '
              f'VAL F1={score:.6f} best={tracker.best_score:.6f} ESB={tracker.stale} '
              f'(observed only) | epoch={epoch_seconds:.1f}s', flush=True)
        progress()
    save_endpoint('fixed_end_12800', payload, validation, predictions, epochs)
    if tracker.virtual_stop_epoch is None:
        save_endpoint('virtual_es15', best_payload, best_validation, best_predictions, tracker.best_epoch)
        endpoints['virtual_es15']['stop_status'] = 'NOT_TRIGGERED_WITHIN_100_EPOCHS'
    else:
        endpoints['virtual_es15']['stop_status'] = 'FIRST_ES15_TRIGGER_FROZEN'
    if model_name == 'local_frozen_projection':
        if not torch.equal(dict(student.named_parameters())[PROJECTION], frozen_reference):
            raise RuntimeError('Fixed projection failed final invariant')
    result = {'status': 'COMPLETED', 'scope': 'MASK_INIT_CROSSED_FULL_BUDGET_VAL_ONLY',
        'model': model_name, 'mask_seed': int(mask_seed), 'init_seed': int(init_seed),
        'epochs_completed': epochs, 'optimizer_steps_completed': epochs*steps,
        'steps_per_epoch': steps, 'endpoints': endpoints, 'epoch_history': history,
        'best_epoch': tracker.best_epoch, 'best_val_image_macro_f1': tracker.best_score,
        'final_epoch_macro_f1': score, 'elapsed_s': time.monotonic()-start,
        'initial_weights_sha256': initial_hash, 'test_access': False,
        'total_revealed_label_presentations': total_presentations,
        'class_label_presentations': class_presentations.tolist(),
        'unique_revealed_labels_seen': {im:int(v.sum()) for im,v in seen.items()},
        'empty_effective_batches': empty_effective, 'empty_physical_microbatches': empty_micro,
        'primary_endpoint': 'fixed_end_12800', 'early_stopping_enabled': False,
        'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
        'checkpoint_sha256': sha(out_dir/'fixed_end_12800.pt'),
        'last_full_state_sha256': sha(out_dir/'last.pt'),
        'effective_config_sha256': sha(out_dir/'effective_config.json')}
    write_json(out_dir/'result.json', result)
    del student, teacher, optimizer
    torch.cuda.empty_cache()
    return result
