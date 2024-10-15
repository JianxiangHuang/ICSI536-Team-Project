import pandas
import numpy


class ProcessDataSet:
    @staticmethod
    def split_dataset_to_traindata_and_testdata(csv_path, test_ratio, random_seed=None):
        dataset = pandas.read_csv(csv_path)

        if random_seed is not None:
            numpy.random.seed(random_seed)

        shuffled_dataset =dataset.sample(frac=1).reset_index(drop=True)

        test_size = int(len(shuffled_dataset) * test_ratio)

        test_set = shuffled_dataset[:test_size].to_numpy()
        train_set = shuffled_dataset[test_size:].to_numpy()

        return train_set, test_set

    @staticmethod
    def print_dataset(dataset,datasetname):
        print("\n"+datasetname+" :")
        print(dataset)
        print("\n")

