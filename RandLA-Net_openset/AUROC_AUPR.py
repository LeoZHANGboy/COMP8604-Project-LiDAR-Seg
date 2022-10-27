import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

# open_label = np.load('/root/autodl-tmp/results/open_labels.npy')
# open_logits = np.load('/root/autodl-tmp/results/open_logits.npy')

close_label = np.load('/root/autodl-tmp/results/close_labels.npy',allow_pickle=True)
close_logits = np.load('/root/autodl-tmp/results/close_logits.npy',allow_pickle=True)

print(close_label)