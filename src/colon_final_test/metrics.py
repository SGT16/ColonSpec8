from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from fewlabel.metrics import segmentation_metrics

BOUNDARY_TOLERANCE_PX = 2.0


def _binary_image(value, name):
    result = np.asarray(value)
    if result.ndim != 2 or not result.size or not np.isin(result, (0, 1)).all():
        raise ValueError(name + "_MUST_BE_NONEMPTY_DENSE_BINARY_2D")
    return result.astype(bool, copy=False)


def dense_boundary_metrics(ytrue, ypred):


    truth = _binary_image(ytrue, "TRUTH")
    predicted = _binary_image(ypred, "PREDICTION")
    if truth.shape != predicted.shape:
        raise ValueError("DENSE_MASK_GEOMETRY_MISMATCH")
    structure = np.ones((3, 3), dtype=bool)
    true_boundary = truth & ~binary_erosion(truth, structure=structure, border_value=0)
    pred_boundary = predicted & ~binary_erosion(predicted, structure=structure, border_value=0)
    n_true, n_pred = int(true_boundary.sum()), int(pred_boundary.sum())
    result = dict(boundary_f1=None, hd95=None, hd95_px=None,
        boundary_precision=None, boundary_recall=None,
        boundary_counts=dict(true=n_true, predicted=n_pred, matched_true=None, matched_predicted=None),
        boundary_definition=dict(tolerance_px=BOUNDARY_TOLERANCE_PX,
            erosion_connectivity=8, outside_image='background',
            extractor='mask_minus_binary_erosion_3x3', distance='euclidean_pixels',
            hd95='maximum_of_two_directional_95th_percentiles', quantile_method='linear'),
        undefined={})
    if not n_true or not n_pred:
        stem = 'BOTH_EMPTY' if not n_true and not n_pred else 'EMPTY_TRUE' if not n_true else 'EMPTY_PRED'
        for name in ('boundary_f1', 'boundary_precision', 'boundary_recall'):
            result['undefined'][name] = stem + '_BOUNDARY'
        result['undefined']['hd95'] = result['undefined']['hd95_px'] = stem + '_MASK'
        return result
    true_to_pred = distance_transform_edt(~pred_boundary)[true_boundary]
    pred_to_true = distance_transform_edt(~true_boundary)[pred_boundary]
    matched_true = int(np.count_nonzero(true_to_pred <= BOUNDARY_TOLERANCE_PX))
    matched_pred = int(np.count_nonzero(pred_to_true <= BOUNDARY_TOLERANCE_PX))
    precision, recall = matched_pred / n_pred, matched_true / n_true
    result.update(boundary_precision=float(precision), boundary_recall=float(recall),
        boundary_f1=float(2*precision*recall/(precision+recall)) if precision+recall else 0.0,
        hd95=float(max(np.percentile(true_to_pred, 95, method='linear'),
                       np.percentile(pred_to_true, 95, method='linear'))))
    result['hd95_px'] = result['hd95']
    result['boundary_counts'].update(matched_true=matched_true, matched_predicted=matched_pred)
    return result


def add_dense_boundary_metrics(metrics, ytrue, ypred):

    value = dict(metrics)
    boundary = dense_boundary_metrics(ytrue, ypred)
    value.update({k: v for k, v in boundary.items() if k != 'undefined'})
    value['undefined'] = dict(metrics.get('undefined', {}), **boundary['undefined'])
    value['not_applicable'] = dict(metrics.get('not_applicable', {}))
    value['metric_status'] = dict(metrics.get('metric_status', {}))
    for name in ('boundary_f1', 'hd95', 'hd95_px', 'boundary_precision', 'boundary_recall'):
        value['not_applicable'].pop(name, None)
        value['metric_status'][name] = 'UNDEFINED' if value[name] is None else 'MEASURED'
    return value


def dense_neural_metrics(ytrue, probabilities):
    truth = _binary_image(ytrue, 'TRUTH')
    probs = np.asarray(probabilities)
    if probs.shape != (*truth.shape, 2):
        raise ValueError('HWC_TWO_CLASS_PROBABILITIES_REQUIRED')
    metrics = segmentation_metrics(truth.reshape(-1).astype(np.int8), probs.reshape(-1, 2))
    labels = probs[:, :, 1] >= .5
    return add_dense_boundary_metrics(metrics, truth, labels)


def dense_gm_metrics(ytrue, scores):
    from fewlabel_gmtssa.worker import score_predictions
    truth = _binary_image(ytrue, 'TRUTH')
    values = np.asarray(scores)
    if values.shape != (*truth.shape, 2) or not np.isfinite(values).all():
        raise ValueError('FINITE_HWC_TWO_CLASS_RESIDUAL_SCORES_REQUIRED')
    metrics, labels, margin = score_predictions(truth.reshape(-1).astype(np.int8), values.reshape(-1, 2))
    labels, margin = labels.reshape(truth.shape), margin.reshape(truth.shape)
    return add_dense_boundary_metrics(metrics, truth, labels), labels, margin


def pooled_boundary_f1(image_metrics):

    counts = dict(true=0, predicted=0, matched_true=0, matched_predicted=0)
    undefined = []
    for image, metrics in image_metrics.items():
        row = metrics['boundary_counts']
        if row['matched_true'] is None or row['matched_predicted'] is None:
            undefined.append(str(image))
            continue
        for key in counts:
            counts[key] += int(row[key])
    result = dict(boundary_f1=None, boundary_counts=counts,
                  total_images=len(image_metrics), valid_images=len(image_metrics)-len(undefined),
                  undefined_images=undefined, status='INCOMPLETE' if undefined else 'COMPLETE')
    if counts['true'] and counts['predicted']:
        p = counts['matched_predicted']/counts['predicted']
        r = counts['matched_true']/counts['true']
        result['boundary_f1'] = float(2*p*r/(p+r)) if p+r else 0.0
    else:
        result['status'] = 'UNDEFINED'
    return result
