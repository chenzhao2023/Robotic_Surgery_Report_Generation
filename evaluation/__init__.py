from .bleu import Bleu
from .rouge import Rouge
from .cider import Cider
import numpy as np

def compute_scores(gts, gen):
    metrics = (Bleu(), Rouge(), Cider())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = np.round(score,4)
        all_scores[str(metric)] = np.round(scores,4)
        
    return all_score, all_scores
