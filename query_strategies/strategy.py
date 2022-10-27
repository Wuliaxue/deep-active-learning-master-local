
class Strategy:
    def __init__(self, dataset, net, n):
        self.dataset = dataset
        self.net = net
        self.n = n
    def query(self, n):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train(self, target_idxs, pattern, i_round, chosenSample_prob):
        labeled_idxs, labeled_data, after_index = self.dataset.get_labeled_data(target_idxs)
        self.net.train(labeled_data, after_index, pattern, i_round, chosenSample_prob)

    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs
    def predice_chosen_prob(self, chosenSample, rd, datasetname):
        prob = self.net.predict_chosen_prob(chosenSample, rd, datasetname)
        return prob
    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings

