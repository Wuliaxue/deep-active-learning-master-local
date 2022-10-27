import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2, help="random seed")
    parser.add_argument('--n_init_labeled', type=int, default=100, help="number of init labeled samples")
    parser.add_argument('--n_query', type=int, default=2000, help="number of queries per round")
    parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
    parser.add_argument('--dataset_name', type=str, default="MNIST",
                        choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10"], help="dataset")
    parser.add_argument('--strategy_name', type=str, default="MarginSampling",
                        choices=["RandomSampling",
                                 "LeastConfidence",
                                 "MarginSampling",
                                 "EntropySampling",
                                 "LeastConfidenceDropout",
                                 "MarginSamplingDropout",
                                 "EntropySamplingDropout",
                                 "KMeansSampling",
                                 "KCenterGreedy",
                                 "BALDDropout",
                                 "AdversarialBIM",
                                 "AdversarialDeepFool"], help="query strategy")

    #attack
    parser.add_argument('--target_num', type=int, default=300, help="number of target samples")

    args = parser.parse_args()
    print(vars(args))

    # fix random seed
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = get_dataset(args.dataset_name)  # load dataset
    net = get_net(args.dataset_name, device)  # load network
    strategy = get_strategy(args.strategy_name)(dataset, net, args.n_query)  # load strategy
    pattern = torch.diag(torch.randn(28))
    # start experiment
    dataset.initialize_labels(args.n_init_labeled)
    print(f"number of labeled pool: {args.n_init_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool - args.n_init_labeled}")
    print(f"number of testing pool: {dataset.n_test}")
    # round 0 accuracy
    print("Round 0")
    target_samples = []
    rd = 0
    strategy.train(target_samples, pattern, rd, chosenSample_prob=[])
    preds = strategy.predict(dataset.get_test_data())
    print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")
    start = time.perf_counter()
    for rd in range(1, args.n_round + 1):
        print(f"Round {rd}")

        # query
        query_idxs = strategy.query()
        target_idxs, chosenSample_prob = strategy.queryTargetSample(args.target_num, rd, args.n_round, args.dataset_name)

        # update labels
        strategy.update(query_idxs)
        strategy.train(target_idxs, pattern, rd, chosenSample_prob)

        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")
    end = time.perf_counter()
    print(round(end - start) / 3600, 'h')
