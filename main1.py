from cfg.option import Options
from train_lstm import lstm_trainer
from preprocess import load_data, load_pretrained_embedding


def main():
    # 读取配置文件
    config_file = './cfg/example.cfg'
    params = Options(config_file)
    train_dataset_path = 'E:\\Bins\\DataSet\\UbiComp2016\\' + params.city.upper() \
                         + 'Data\\Checkins-Sequence-filter-leq5-spilt82-train-' + params.city.upper() + '.csv'
    category_sequence, category2id = load_data(train_dataset_path)

    model_name = 'pte_g'
    pretrained_embedding_path = './output/' + model_name + '@100_tky.txt'
    pretrained_embedding = load_pretrained_embedding(pretrained_embedding_path)

    # print(save_path)
    lstm_trainer(category_sequence, category2id, pretrained_embedding, params, model_name)


if __name__ == '__main__':
    main()
