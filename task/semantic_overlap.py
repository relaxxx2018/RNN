import numpy as np

# category层次结构树
category_path = 'E:\\JupyterNoteBookSpace\\Hierarchical Tree\\data\\category.csv'


def get_category_to_type():
    category_to_type = {}
    # Arts & Entertainment/College & University/Food/
    # Nightlife Spot/Outdoors & Recreation/Professional & Other Places/
    # Residence/Shop & Service/Travel & Transport
    category_to_id = {'4d4b7104d754a06370d81259': 0, '4d4b7105d754a06372d81259': 1, '4d4b7105d754a06374d81259': 2,
                      '4d4b7105d754a06376d81259': 3, '4d4b7105d754a06377d81259': 4, '4d4b7105d754a06375d81259': 5,
                      '4e67e38e036454776db1fb3a': 6, '4d4b7105d754a06378d81259': 7, '4d4b7105d754a06379d81259': 8,
                      '4d4b7105d754a06373d81259': 9}

    with open(category_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    for line in lines:

        temp = line.split(',')
        category_name, category_id, category_type = temp[1], temp[0], temp[2]

        if '$' not in category_type:
            category_to_type[category_name] = category_to_id[category_id]
        else:
            temp = category_type.split('$')
            category_to_type[category_name] = category_to_id[temp[len(temp) - 2]]
    return category_to_type


def semantic_overlap(category_embedding_path, category_to_type, n):
    category_embedding = {}

    f = open(category_embedding_path, 'r', encoding='utf-8')
    line = f.readline()
    while line != '':
        temp = line.strip().split(',')
        category = temp[0]
        embedding = [float(temp[i]) for i in range(1, len(temp))]
        category_embedding[category] = embedding

        line = f.readline()

    mean_match_rate, count = 0, len(category_embedding.keys())
    for current_category in category_embedding.keys():
        rank_list = {}
        current_embedding = category_embedding.get(current_category)
        for candidate_category in category_embedding.keys():
            if current_category == candidate_category: continue
            candidate_embedding = category_embedding.get(candidate_category)
            cos_sim = get_cosine_similarity(current_embedding, candidate_embedding)
            rank_list.update({candidate_category: cos_sim})
        rank_list = sorted(rank_list.items(), key=lambda x: x[1], reverse=False)
        category_rank = [t[0] for t in rank_list][-n:]
        match_count = 0
        for candidate_category in category_rank:
            if category_to_type.get(current_category) == category_to_type.get(candidate_category):
                match_count += 1

        match_rate = match_count / n
        mean_match_rate += match_rate
    mean_match_rate = mean_match_rate / count

    return mean_match_rate


def get_cosine_similarity(vector1, vector2):
    v1, v2 = np.array(vector1), np.array(vector2)
    sim = np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))
    return sim


def main():
    category_to_type = get_category_to_type()

    semantic_overlap_k = [1, 5, 10]
    # semantic_overlap_k = [1, 5, 10, 20, 30, 40, 50]
    city, filename = 'tky', 'lstm'
    vector_sizes = range(100, 110, 20)

    for vector_size in vector_sizes:
        print('vector size:' + str(vector_size))
        mrr2_list = []
        for k in semantic_overlap_k:
            category_embedding_path = '../output/' + filename + '@' + str(vector_size) + '_' + city + '.txt'
            mrr2 = semantic_overlap(category_embedding_path, category_to_type, k)
            mrr2_list.append(str(format(mrr2, '.3f')))
        print(','.join(mrr2_list))


if __name__ == '__main__':
    main()
