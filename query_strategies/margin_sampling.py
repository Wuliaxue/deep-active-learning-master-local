import numpy as np
import torch

from .strategy import Strategy

class MarginSampling(Strategy):
    def __init__(self, dataset, net, n):
        super(MarginSampling, self).__init__(dataset, net, n)

    def query(self):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]
        return unlabeled_idxs[uncertainties.sort()[1][:self.n]]

    def queryTargetSample(self, target_num):
        unlabeled_idxs = self.query()
        unlabeled_data = self.dataset.get_unlabeled_data_by_index(unlabeled_idxs)
        probs = self.predict_prob(unlabeled_data)
        centerVector = probs[int((self.n / 2))]
        temp = torch.pairwise_distance(probs, centerVector)
        temp_sorted, idxs = temp.sort()
        return unlabeled_idxs[idxs[:target_num]]


