import numpy as np
import torch
import utils
from .strategy import Strategy

class MarginSampling(Strategy):
    def __init__(self, dataset, net, n):
        super(MarginSampling, self).__init__(dataset, net, n)
        self.chosenSample_prob = []
        self.chosenSample = []
        self.querySamples_idxs = []
    def query(self):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data)
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]
        self.querySamples_idxs = unlabeled_idxs[uncertainties.sort()[1][:self.n]]
        return unlabeled_idxs[uncertainties.sort()[1][:self.n]]

    def queryTargetSample(self, target_num, rd, n_round, datasetname):
        # unlabeled_idxs = self.query()
        unlabeled_data = self.dataset.get_unlabeled_data_by_index(self.querySamples_idxs)
        probs = self.predict_prob(unlabeled_data)
        if rd == 1:
            self.chosenSample = self.dataset.get_unlabeled_data_by_index(self.querySamples_idxs[int(self.n / 2)])
        # print(self.chosenSample.X)
        self.chosenSample_prob = self.predice_chosen_prob(self.chosenSample, rd, datasetname)
        print(self.chosenSample_prob)
        # temp = torch.pairwise_distance(probs, self.chosenSample_prob)
        # if rd == n_round:
        #     temp_sorted, idxs = temp.sort(descending=True)
        # else:
        #     temp_sorted, idxs = temp.sort()
        # return self.querySamples_idxs[idxs[:target_num]]
        idxs = np.where(torch.max(probs, 1)[1].cpu() == torch.max(self.chosenSample_prob, 1)[1].cpu())
        if rd  >= 9:
            idxs = np.where(torch.max(probs, 1)[1] == torch.max(self.chosenSample_prob, 1)[1] + 1)
        return self.querySamples_idxs[idxs], self.chosenSample_prob



