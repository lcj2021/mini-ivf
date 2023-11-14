from vector_io import *
import numpy as np

# query = fvecs_read("/RF/dataset/sift/sift1m_pq128_100k_kc1000_seg20_sample/train_query.fvecs")
# sample = ivecs_read("/RF/dataset/sift/sift1m_pq128_100k_kc1000_seg20_sample/train_sample.ivecs")
# radius = ivecs_read("/RF/dataset/sift/sift1m_pq128_100k_kc1000_seg20_sample/train_radius.ivecs")
# distance = fvecs_read("/RF/dataset/sift/sift1m_pq128_100k_kc1000_seg20_sample/train_distance.fvecs")
posting_list_size = ivecs_read("/dk/anns/dataset/sift10m/nt1m_pq64_kc16384/posting_lists_lens.ulvecs")

print(len(posting_list_size))

# query = fvecs_read("/RF/dataset/sift/sift1m_pq128_100k_kc1000_seg20_sample/tuning_query.fvecs")
# sample = ivecs_read("/RF/dataset/sift/sift1m_pq128_100k_kc1000_seg20_sample/tuning_sample.ivecs")
# radius = ivecs_read("/RF/dataset/sift/sift1m_pq128_100k_kc1000_seg20_sample/tuning_radius.ivecs")
# distance = fvecs_read("/RF/dataset/sift/sift1m_pq128_100k_kc1000_seg20_sample/tuning_distance.fvecs")

# pred = ivecs_read("/RF/dataset/sift/sift1m_pq128_100k_kc1000_seg20_sample/pred.ivecs")

# print(sample[0][0:1000])
# print(np.count_nonzero(sample[0][0:1000]))
# print(radius[0][0:1000])
# print(np.count_nonzero(radius[0][0:1000]))
# print(distance[0][0:1000])
# print(query[0][0:])
# print(query.shape)
# print(sample.shape)
# print(radius[0])
# print("pred: \n", pred[0])

# train_cnt4cluster = np.zeros(1000)
# print(radius.shape)
# for i in range(radius.shape[0]):
#     for j in range(radius.shape[1]):
#         if radius[i][j] != 0:
#             train_cnt4cluster[j] += 1

# print(train_cnt4cluster)
