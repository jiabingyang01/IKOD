import torch
score_sum = torch.tensor([[[0.4, 0.2, 0.3, 0.1, 0.5, 0.6, 1.0, 0.8, 0.9, 0.7], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1]]]).cuda()
score_idx = torch.zeros(2, 10).cuda()

for i in range(score_sum.shape[1]):
    att_sum = 0
    for j in range(score_sum.shape[-1]):
        att_sum += score_sum[0, i, j]
        score_idx[i, j] = att_sum

# score_idx[:, :10] = score_sum.mean(dim= -1, keepdim=False)
print(score_idx)
# print(score_sum[0, 4:8])
# print(torch.argsort(score_sum[0, 4:8]))
# print(torch.where(torch.argsort(score_sum[0, 4:8])>2)[0])
# print(torch.where(torch.argsort(score_sum[0, 4:8])<=2)[0])

# a = [1, 2, 3, 4]
# print(a[1:3])

# print(torch.range(1, 4))