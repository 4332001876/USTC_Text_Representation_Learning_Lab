import struct
import os
import numpy as np
from tqdm import tqdm

gt_file = '../../data/wordsim_similarity_goldstandard.txt'

def read_vec_bin(file):
    vectors = {}
    with open(file, 'rb') as f:
        c = None
        # read the first line: the number of words and the dimension of the vectors
        num = b''
        c = f.read(1)
        while c != b' ':
            num += c
            c = f.read(1)
        num_words = int(num)

        num = b''
        c = f.read(1)
        while c != b'\n':
            num += c
            c = f.read(1)
        dim = int(num)
        
        for _ in tqdm(range(num_words)):
            word = b''
            c = f.read(1)
            while c != b' ':
                word += c
                c = f.read(1)
            word = word.decode('utf-8').lower()
            vec = []
            for i in range(dim):
                vec.append(struct.unpack('f', f.read(4))[0])
            vectors[word] = np.array(vec)
            _ = f.read(1) # \n
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
            if words[0].lower() not in vectors or words[1].lower() not in vectors:
                print(f'{words[0]} or {words[1]} not in the vocabulary')
                continue
            v1 = vectors[words[0].lower()]
            v2 = vectors[words[1].lower()]
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
    vec_files = ['vectors_hs_cbow.bin', 'vectors_hs_sg.bin', 'vectors_ns_cbow.bin', 'vectors_ns_sg.bin']
    for vec_file in vec_files:
        exp(vec_file)
    