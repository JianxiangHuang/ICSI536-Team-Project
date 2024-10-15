import pandas
import numpy


# this class contain methods used to process the dataset
class ProcessDataSet:
    # this method is used to split train dataset and test dataset
    # csv_path is the path to the dataset
    # test_ratio is the ratio decide the proportion of the dataset to be used as test set
    # random_seed is used to control the random split process, default is None which is random everytime, but if it fixed, this method will generate same result everytime
    @staticmethod
    def split_dataset_to_traindata_and_testdata(csv_path, test_ratio=0.1, random_seed=None):
        dataset = pandas.read_csv(csv_path)

        if random_seed is not None:
            numpy.random.seed(random_seed)

        shuffled_dataset = dataset.sample(frac=1).reset_index(drop=True)

        test_size = int(len(shuffled_dataset) * test_ratio)

        test_set = shuffled_dataset[:test_size].to_numpy()
        train_set = shuffled_dataset[test_size:].to_numpy()
        return train_set, test_set

    # this method print the dataset inputted as parameter
    @staticmethod
    def print_dataset(dataset, datasetname):
        print("\n" + datasetname + " :")
        print(dataset)
        print("\n")
