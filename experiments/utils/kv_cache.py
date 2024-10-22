import torch
import random
import numpy as np
import bisect
def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]

import torch.nn.functional as F
DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

class ElasticCache:
    def __init__(
        self,
        start_size=4,
        recent_size=2048,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.5,
        distance=-25,
        layer_num=40,
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

        self.score_idx = torch.zeros(layer_num, self.cache_size + 1).cuda()
        self.ratio = ratio
        self.protect_size = 1
        self.flag = True
        self.distance = distance
        self.layer_num = layer_num

        self.selected_idx = 0
        
    def __call__(self, past_key_values, num_of_token=None, attentions=None, token_position=None):
        self.start_size = token_position['image_end']
        # self.start_size = 0
        # print("num_of_token: ", num_of_token)
        if past_key_values is None:
            return None
        attn_score = [attention for attention in attentions]
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        # update attn score 
        # print(torch.cat(attn_score, dim=0).shape)
        # attn_score = torch.cat(attn_score, dim=0)[...,token_position['image_start']:token_position['image_end']]
        # attn_score = torch.cat(attn_score, dim=0)
        # # print(attn_score.shape)
        # attn_score = attn_score.mean(dim=1, keepdim=False).mean(dim=1, keepdim=False)   
        # print(len(attn_score))

        # attn_score = torch.cat(attn_score, dim=0)
        attn_score = torch.cat(attn_score, dim=0)[...,token_position['image_start']:token_position['image_end']]
        attn_score = attn_score.mean(dim=1, keepdim=False) 
        # self.score_idx[:, :seq_len] = attn_score[:, -1, :].squeeze()
        # print(self.score_idx[15,self.start_size:seq_len])
        # print(attn_score[15,-1,self.start_size:seq_len])

        if attn_score.shape[-2] > 1:
            self.score_idx[:, :num_of_token] = attn_score.sum(dim=-1, keepdim=False)
        else:
            self.score_idx[:, num_of_token-1] = attn_score.mean(dim=1, keepdim=False).sum(dim=1, keepdim=False)

        # if attn_score.shape[-2] > 1:
        #     # assert self.flag is True # only use for the first time
        #     for idx in range(attn_score.shape[-1]):
        #         cur_score = attn_score[:, idx, :idx+1]
        #         self.score_idx[:, :(cur_score.shape[-1])] += cur_score
        # else:
        #     self.score_idx[:, :attn_score.shape[-1]] += attn_score[:,0,:attn_score.shape[-1]]        

        # if attn_score.shape[-1] > 0:
        #     # assert self.flag is True # only use for the first time
        #     # for idx in range(seq_len):
        #     #     self.score_idx[:, :(attn_score.shape[-1])] += attn_score
        #     self.score_idx[:, num_of_token-3] = attn_score.mean(dim=1)
        # else:
        #     pass
        # print(self.score_idx[15, 611:num_of_token])
        # print(self.start_size, num_of_token)

        # forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        select_num = int(self.ratio * (num_of_token - self.start_size - self.protect_size))
        if select_num <= 0:
            return past_key_values
        else:
        # if forget_num > 1:
            # assert self.flag is True
            self.flag = False

            selected_idx_all = []
            merge_idx_all = []
            throw_idx_all = []

            for idx in range(self.layer_num):
                selected_idx = torch.where(torch.argsort(self.score_idx[idx, self.start_size:(seq_len - self.protect_size)]) <= select_num-1)[0] + self.start_size                
                throw_idx = torch.where(torch.argsort(self.score_idx[idx, self.start_size:(seq_len - self.protect_size)]) > select_num-1)[0]
                
                # selected_idx = torch.arange(seq_len-self.protect_size-select_num, seq_len - self.protect_size).cuda()
                # throw_idx = torch.arange(self.start_size, seq_len-self.protect_size-select_num).cuda()
                 
                # print("score_idx: ", self.score_idx[idx, self.start_size:(seq_len - self.protect_size)])


                # # Generate all indices in the range [self.start_size, seq_len - self.protect_size)
                # all_idx = torch.arange(self.start_size, seq_len - self.protect_size).cuda()

                # # Shuffle the indices randomly
                # shuffled_idx = all_idx[torch.randperm(len(all_idx)).cuda()]

                # # Randomly select the first select_num indices as selected_idx
                # selected_idx = shuffled_idx[:select_num]

                # # The remaining indices are considered throw_idx
                # throw_idx = shuffled_idx[select_num:]


                if selected_idx.numel() == 0 or throw_idx.numel() == 0:
                    return past_key_values                 

      
                merge_idx = []

                # for i in range(len(throw_idx)):
                #     merge_idx.append(selected_idx[torch.abs((selected_idx - throw_idx[i])).argmin()].unsqueeze(0)) 
                # merge_idx = torch.cat(merge_idx)

                # bound_idx = []
                # window_idx = [0 for i in range(num_of_token+10)]                                
                # for i in range(len(selected_idx)):
                #     window_idx[selected_idx[i]] = 1
                # for i in range(len(selected_idx)-1):
                #     bound_idx.append((selected_idx[i]+selected_idx[i+1])/2)
                # bound_idx.append(seq_len - self.protect_size) 
                # j = 0
                # for i in range(self.start_size, seq_len - self.protect_size):                                    
                #     if i > bound_idx[j]:
                #         j += 1
                #     if window_idx[i] == 0 and i <= bound_idx[j]:
                #         merge_idx.append(selected_idx[j].unsqueeze(0))   
                 

                # Calculate the difference between each throw_idx and selected_idx
                # Use broadcasting to compute the absolute difference between all throw_idx and all selected_idx
                diff = torch.abs(selected_idx.unsqueeze(1) - throw_idx.unsqueeze(0))

                # Find the closest selected_idx for each throw_idx
                min_indices = torch.argmin(diff, dim=0)

                # Extract the closest selected_idx
                merge_idx = selected_idx[min_indices]             


                # print(merge_idx)


                merge_idx_all.append(merge_idx)
                throw_idx_all.append(throw_idx)
                selected_idx = torch.cat([torch.arange(self.start_size).cuda(), selected_idx, torch.arange(seq_len - self.protect_size,seq_len).cuda()], dim=0) # the last token is always kept
                selected_idx_all.append(selected_idx)                
            # print("selected_idx_all: ", selected_idx_all)
            # print("throw_idx_all: ", throw_idx_all)

            past_key_values_return = []
            # c = 0
            # for idx, (k, v) in enumerate(past_key_values):
            #     c = c + 1
            #print(c)
            for idx, (k, v) in enumerate(past_key_values):
                selected_idx = selected_idx_all[idx]
                merge_idx = merge_idx_all[idx]
                throw_idx = throw_idx_all[idx]

                k_forget = k.gather(dim=-2, index=throw_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]))
                v_forget = v.gather(dim=-2, index=throw_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]))
                # k_forget_mean = k_forget.mean(dim=-2, keepdim=True).expand(k.shape[0], k.shape[1], merge_idx.shape[-1] ,k.shape[-1])
                # v_forget_mean = v_forget.mean(dim=-2, keepdim=True).expand(v.shape[0], v.shape[1], merge_idx.shape[-1] ,v.shape[-1])

                # print("k_mean: ", k_mean[:,:,:5,:])
                k = k.scatter_reduce(-2, merge_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]), k_forget, 'mean')
                v = v.scatter_reduce(-2, merge_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]), v_forget, 'mean')
                # print("k: ", k.shape)
                k_new = k.gather(dim=-2, index=selected_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]))
                v_new = v.gather(dim=-2, index=selected_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]))
                # print("k_new: ", k_new.shape)
                past_key_values_return.append([k_new, v_new])
            return past_key_values_return
            # else:
            #     selected_idx = self.selected_idx
            #     return [[torch.cat([self.k_slice(k, 0, selected_idx), self.k_slice(k, (selected_idx+1), seq_len),],
            #                 dim=self.k_seq_dim,),
            #             torch.cat([self.v_slice(v, 0, selected_idx), self.v_slice(v, (selected_idx+1), seq_len),],
            #                 dim=self.v_seq_dim,)]
            #         for k, v in past_key_values]
            

class LocalCache:
    def __init__(
        self,
        start_size=4,
        recent_size=1024,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.ratio = ratio

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values
        else:
            return [[torch.cat([self.k_slice(k, 0, self.start_size), self.k_slice(k, forget_num + self.start_size, seq_len),],
                        dim=self.k_seq_dim,),
                    torch.cat([self.v_slice(v, 0, self.start_size), self.v_slice(v, forget_num + self.start_size, seq_len),],
                        dim=self.v_seq_dim,),]
                for k, v in past_key_values]
        

class H2OCache:
    def __init__(
        self,
        start_size=4,
        recent_size=1024,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

        self.score_idx = torch.zeros(self.cache_size + 1).cuda()
        self.ratio = ratio
        self.protect_size = 1
        self.flag = True

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        attn_score = [attention for attention in attentions]
        past_key_values_new = tuple(x for x in past_key_values)
        seq_len = past_key_values_new[0][0].size(self.k_seq_dim)
        past_key_values_new = tuple(
            [x[:, :, :-1, :] for x in layer] for layer in past_key_values
        )

        if attn_score.shape[-2] > 1:
            assert self.flag is True # only use for the first time
            for idx in range(attn_score.shape[-1]):
                cur_score = attn_score[idx][:idx+1]
                self.score_idx[:len(cur_score)] += cur_score
        else:
            attn_score = attn_score.squeeze(0)
            self.score_idx[:seq_len] += attn_score

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        self.protect_size = 3
        if forget_num <= 0:
            return past_key_values_new
        else:
            if forget_num > 1:
                assert self.flag is True
                self.flag = False
                selected_idx = torch.where(torch.argsort(self.score_idx[:(seq_len - self.protect_size)]) > forget_num)[0]
                selected_idx = torch.cat([selected_idx, torch.arange(seq_len - self.protect_size, seq_len).cuda()], dim=0)
                past_key_values_return = []
                for k, v in past_key_values_new:
                    k_new = k.gather(dim=-2, index=selected_idx.view(1,1,-1,1).expand(k.shape[0], k.shape[1], -1 ,k.shape[-1]))
                    v_new = v.gather(dim=-2, index=selected_idx.view(1,1,-1,1).expand(v.shape[0], v.shape[1], -1 ,v.shape[-1]))
                    past_key_values_return.append([k_new, v_new])
                
                return past_key_values_return
            else:
                selected_idx = self.score_idx[self.start_size:(seq_len - self.protect_size)].argmin() + self.start_size
                self.score_idx[(selected_idx):-1] = self.score_idx[(selected_idx+1):].clone()
                
                return [[torch.cat([self.k_slice(k, 0, selected_idx), self.k_slice(k, (selected_idx+1), seq_len),],
                            dim=self.k_seq_dim,),
                        torch.cat([self.v_slice(v, 0, selected_idx), self.v_slice(v, (selected_idx+1), seq_len),],
                            dim=self.v_seq_dim,)]
                    for k, v in past_key_values_new]

