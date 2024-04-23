# size=200    window=8    min-count=5


# hs_cbow

time ./word2vec -train ../../data/text8 -output vectors_hs_cbow.bin \
-cbow 1 -size 200 -window 8 -negative 0 -hs 1 -sample 1e-4 -threads 20 -binary 1 -iter 15 -min-count 5


# hs_sg

time ./word2vec -train ../../data/text8 -output vectors_hs_sg.bin \
-cbow 0 -size 200 -window 8 -negative 0 -hs 1 -sample 1e-4 -threads 20 -binary 1 -iter 15 -min-count 5


# ns_cbow

time ./word2vec -train ../../data/text8 -output vectors_ns_cbow.bin \
-cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15 -min-count 5


# ns_sg

time ./word2vec -train ../../data/text8 -output vectors_ns_sg.bin \
-cbow 0 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15 -min-count 5


