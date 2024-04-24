import struct
import os
import numpy as np

gt_file = '../../data/wordsim_similarity_goldstandard.txt'

def read_vec_bin(file):
    vectors = {}
    with open(file, 'rb') as f:
        c = None
        # read the first line: the number of words and the dimension of the vectors
        num = b''
        while (c := f.read(1)) != b' ':
            num += c
        num_words = int(num)

        num = b''
        while (c := f.read(1)) != b'\n':
            num += c
        dim = int(num)
        
        while c != b'':
            word = b''
            while (c := f.read(1)) != b' ':
                word += c
            word = word.decode('utf-8')
            vec = []
            for i in range(dim):
                vec.append(struct.unpack('f', f.read(4))[0])
            vectors[word] = np.array(vec)
    return vectors

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def evaluate(vectors, file):
    ground_truth_cosine_similarity = []
    result_cosine_similarity = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.split()
            if words[0] not in vectors or words[1] not in vectors:
                print(f'{words[0]} or {words[1]} not in the vocabulary')
                continue
            v1 = vectors[words[0]]
            v2 = vectors[words[1]]
            ground_truth_cosine_similarity.append(float(words[2]))
            result_cosine_similarity.append(cosine_similarity(v1, v2))
    ground_truth_cosine_similarity = np.array(ground_truth_cosine_similarity)
    result_cosine_similarity = np.array(result_cosine_similarity)
    # return Spearman's rank correlation coefficient
    gt_rank = ground_truth_cosine_similarity.argsort().argsort()
    result_rank = result_cosine_similarity.argsort().argsort()
    return np.corrcoef(gt_rank, result_rank)[0, 1]
    
def exp(vec_file):
    print(vec_file)
    vectors = read_vec_bin(vec_file)
    print(evaluate(vectors, gt_file))

if __name__ == '__main__':
    vec_files = ['../../data/vec_50.bin', '../../data/vec_100.bin', '../../data/vec_300.bin']
    for vec_file in vec_files:
        exp(vec_file)
    