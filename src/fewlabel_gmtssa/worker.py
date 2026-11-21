def score_predictions(truth,scores):

    import numpy as np
    from fewlabel.metrics import segmentation_metrics
    from sklearn.metrics import roc_auc_score,average_precision_score
    labels=np.argmax(scores,axis=1)
    metrics=segmentation_metrics(truth,np.eye(2)[labels])
    margin=scores[:,1]-scores[:,0]
    if len(np.unique(truth))==2:
        metrics['auroc']=float(roc_auc_score(truth,margin))
        metrics['average_precision']=float(average_precision_score(truth,margin))
    for key in ('brier','brier_score','ece','ece15'):
        metrics[key]=None; metrics['not_applicable'][key]='NO_PROBABILITY_MODEL_DISTANCE_CLASSIFIER'
        metrics['metric_status'][key]='NOT_APPLICABLE'
    metrics['calibration_bins']=[]
    metrics.update(threshold=None,tie_prediction=0,prediction_rule='argmax_negative_squared_residual',
        ranking_score='residual_non_tumor_squared_minus_residual_tumor_squared',probabilities_reported=False)
    return metrics,labels,margin
