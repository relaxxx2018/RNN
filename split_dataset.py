import random
from tqdm import tqdm


def split_dataset(city, percentage):
    checkin_sequence_dir = 'E:\\Bins\\DataSet\\UbiComp2016\\' + city.upper() \
                           + 'Data\\Checkins-Sequence-filter-leq5-' + city.upper() + '.csv'

    with open(checkin_sequence_dir, 'r', encoding='utf-8') as f:
        transition_sequences = f.readlines()

    train_dataset = []
    test_dataset = []
    random.seed(1)
    for sequence in tqdm(transition_sequences):
        prob = random.random()
        if prob > percentage:
            train_dataset.append(sequence)
        else:
            test_dataset.append(sequence)

    f = open(
        'E:\\Bins\\DataSet\\UbiComp2016\\' + city.upper() \
        + 'Data\\Checkins-Sequence-filter-leq5-spilt82-train-' + city.upper() + '.csv', 'w',
        encoding='utf-8')
    for i in train_dataset:
        f.write(i)
    f.close()

    f = open(
        'E:\\Bins\\DataSet\\UbiComp2016\\' + city.upper() \
        + 'Data\\Checkins-Sequence-filter-leq5-spilt82-test-' + city.upper() + '.csv', 'w',
        encoding='utf-8')
    for i in test_dataset:
        f.write(i)
    f.close()


if __name__ == '__main__':
    split_dataset('tky', 0.2)
