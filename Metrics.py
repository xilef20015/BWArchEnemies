import numpy as np

from sklearn.metrics import make_scorer

def ranked_probability_score(y_true, y_pred):
    """Returns the ranked probability score (RPS) for a single match

    Args:
        y_probs (sequential): 1d tuple/array/list containing the probabilities for possible outcomes
        y (int): integer number for possible classes/outcomes

    Returns:
        float: ranked probability score
    """
        
    r = len(y_pred)
    
    a = ()
    match y_true:
        case 0:
            a = (1,0,0)
        case 1:
            a = (0,1,0)
        case 2:
            a = (0,0,1)
    
    rps = 0
    for i in range(r-1):
        rps_term = 0
        for j in range(i+1):
            rps_term += y_pred[j]-a[j]
        
        rps += np.square(rps_term)
        
    return rps/(r-1)

def avg_ranked_probability_score(y_true, y_pred_prob):
    """Returns the average rps

    Args:
        y_pred_prob (2d array): 2d array/list containing probabilities for outcomes of every input
        y_true (1d array): true outcome

    Returns:
        float: mean rps (rps/n)
    """
    n= len(y_true)
    
    avg_rps = 0
    
    for idx, y in enumerate(y_true):
        avg_rps += ranked_probability_score(y, y_pred_prob[idx])/n
    
    return avg_rps

if __name__ == "__main__":
    
    print(ranked_probability_score((0.9, 0.05, 0.05), 0))