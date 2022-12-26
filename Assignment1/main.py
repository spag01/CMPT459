import pandas as pd
import argparse
from random_forest import RandomForest

def parse_args():
    parser = argparse.ArgumentParser(description='Run random forrest with specified input arguments')
    parser.add_argument('--n-classifiers', type=int,
                        help='number of features to use in a tree',
                        default=1)
    parser.add_argument('--train-data', type=str, default='data/train.csv',
                        help='train data path')
    parser.add_argument('--test-data', type=str, default='data/test.csv',
                        help='test data path')
    parser.add_argument('--criterion', type=str, default='entropy',
                        help='criterion to use to split nodes. Should be either gini or entropy.')
    parser.add_argument('--maxdepth', type=int, help='maximum depth of the tree',
                        default=5)
    parser.add_argument('--min-sample-split', type=int, help='The minimum number of samples required to be at a leaf node',
                        default=20)
    parser.add_argument('--max-features', type=int,
                        help='number of features to use in a tree',
                        default=12)
    parser.add_argument('--run-task4', type=bool,
                        help='run assignment task 4 and show the plot',
                        default=False)
    a = parser.parse_args()
    return(a.n_classifiers, a.train_data, a.test_data, a.criterion, a.maxdepth, a.min_sample_split, a.max_features, a.run_task4)


def read_data(path):
    data = pd.read_csv(path, skipinitialspace=True)
    return data

def main():
    n_classifiers, train_data_path, test_data_path, criterion, max_depth, min_sample_split, max_features, run_task4 = parse_args()
    train_data = read_data(train_data_path)
    test_data = read_data(test_data_path)

    train_data.replace({r'\s*\?\s*': 'None' }, inplace=True, regex=True)
    test_data.replace({r'\s*\?\s*': 'None', '50K.': '50K'}, inplace=True, regex=True)
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)


    if run_task4:
        import matplotlib.pyplot as plt
        import numpy as np
        test_scores = []
        train_scores = []
        for max_depth in range(1, 11):

            random_forest = RandomForest(n_classifiers=10,
                  criterion = 'gini',
                  max_depth=  max_depth,
                  min_samples_split = 10 ,
                  max_features = 10 )
            
            train_score = random_forest.fit(train_data, 'income')
            test_score = random_forest.evaluate(test_data, 'income')
            train_scores.append(train_score)
            test_scores.append(test_score)
            print('\nMax depth :',max_depth)
            print(f'Train accuracy:{train_score} Test accuracy:{test_score}')

        fig, ax = plt.subplots()
        ax.plot(np.arange(1, 11), test_scores)
        ax.plot(np.arange(1, 11), train_scores)
        ax.set_xlabel('Max Depth')
        ax.set_ylabel('Accuracy')
        plt.legend(['Test', 'Train'], loc='upper left')
        ax.set_title('Random Forest Accuracy vs Max Depth')
        plt.savefig('Question4.png')
        
        max_depth = list(range(1,11))
        zipped = list(zip(max_depth, train_scores, test_scores))
        df = pd.DataFrame(zipped, columns=['max_depth', 'train_scores', 'test_scores']) 
        task4_csv_data = df.to_csv('task4.csv', index = False)

    else:
        random_forest = RandomForest(n_classifiers=n_classifiers,
                  criterion = criterion,
                  max_depth=  max_depth,
                  min_samples_split = min_sample_split ,
                  max_features = max_features )

        print(random_forest.fit(train_data, 'income'))
        print(random_forest.evaluate(train_data, 'income'))
        print(random_forest.evaluate(test_data, 'income'))

        

if __name__ == '__main__':
    main()

