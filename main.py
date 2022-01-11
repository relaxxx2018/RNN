from cfg.option import Options
from train_rnn import lstm_trainer


def load_data(path):
    """
    :param path: check-ins sequence
    :return:
        category: [[category_id, category_id, ...], [], ...]
        category2id: {category: category_id, ...}
    """
    print('loading train data .......')
    # venue_sequence = []
    category_sequence = []
    category2id = dict()

    f = open(path, 'r', encoding='UTF-8')
    content = f.readline()
    while content != '':
        items = content.strip().split(',')

        temp1, temp2 = [], []
        for item in items[1:]:
            category = item[item.index('#') + 1: item.index('@')]
            # venue = item[: item.index('#')]
            if category not in category2id.keys():
                category2id[category] = len(category2id.keys())
            temp1.append(category2id[category])
            # temp2.append(venue)
        category_sequence.extend(temp1)
        # venue_sequence.append(temp2)
        content = f.readline()
    f.close()

    return category_sequence, category2id


def main():
    # 读取配置文件
    config_file = './cfg/example.cfg'
    params = Options(config_file)
    sequence_info_path = 'E:\\Bins\\DataSet\\UbiComp2016\\' + params.city.upper() \
                         + 'Data\\Checkins-Sequence-filter-leq5-' + params.city.upper() + '.csv'
    category_sequence, category2id = load_data(sequence_info_path)

    lstm_trainer(category_sequence, category2id, params)


if __name__ == '__main__':
    main()