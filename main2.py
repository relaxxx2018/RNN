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
    pretrained_embedding_path = './output/sc@100_tky.txt'
    pretrained_embedding = load_pretrained_embedding(pretrained_embedding_path)
    save_path = params.embedding_output_path + '/sc@' + str(params.init_embed_size) + '_' + params.city + '.pth'
    print(save_path)
    lstm_trainer(category_sequence, category2id, pretrained_embedding, params, save_path)


if __name__ == '__main__':
    main()
