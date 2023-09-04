from __future__ import absolute_import, print_function
import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable


class MemoryUnit(nn.Module):
    def __init__(self, ptt_num, num_cls, part_num, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        '''
        the instance PTT is divided into cls_number x ptt_number per cls x part number per ptt
        '''
        self.num_cls = num_cls
        self.ptt_num = ptt_num
        self.part_num = part_num

        self.mem_dim = ptt_num * num_cls * part_num  # M
        self.fea_dim = fea_dim  # C
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        # self.sem_weight = Parameter(torch.Tensor(self.num_cls, self.fea_dim)) # N x C
        self.bias = None
        self.shrink_thres = shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.reweight_layer_part = nn.Conv1d(self.part_num, self.part_num, 1)
        self.reweight_layer_ins = nn.Conv1d(self.ptt_num, self.ptt_num, 1)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def get_update_query(self, mem, max_indices, score, query):
        m, d = mem.size()

        query_update = torch.zeros((m, d)).cuda()
        # random_update = torch.zeros((m,d)).cuda()
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1) == i)
            a, _ = idx.size()
            # ex = update_indices[0][i]
            if a != 0:
                # random_idx = torch.randperm(a)[0]
                # idx = idx[idx != ex]
                #                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)), dim=0)
                # random_update[i] = query[random_idx] * (score[random_idx,i] / torch.max(score[:,i]))
            else:
                query_update[i] = 0
                # random_update[i] = 0

        return query_update

    def forward(self, input, residual=False):
        '''
        this is a bottom-up hierarchical stastic and summaration module
        all steps in main flow follow  part -> prototype -> cls
        input = NHW x C
        total PTT M =  num_cls (L) x ptt_num (T) x part_num (P)
        dimension C = fea_dim
        '''
        ### for global part-unware instance PTT, act as sub flow
        #print("input = ",input.shape)
        att_weight = F.linear(input,self.weight)
        # we doesn't split the part dimension, there it is part-unaware NHW x M
        #print("att_weight = ",att_weight)
        import pdb
        # pdb.set_trace()
        att_weight = F.softmax(att_weight, dim=1)  # NHW x M
        ### update ###
        # _, gather_indice = torch.topk(att_weight, 1, dim=1)
        # ins_mem_sample_driven = self.get_update_query(self.weight, gather_indice, att_weight,input)
        # self.weight.data = F.normalize(ins_mem_sample_driven+ self.weight, dim=1)

        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output_part = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC

        ### for global part-aware instance PTT

        self.reweight_part = self.weight.view(self.num_cls * self.ptt_num, self.fea_dim, -1).permute(0, 2, 1)
        self.reweight_part = (torch.sigmoid(self.reweight_layer_part(self.reweight_part)) * self.reweight_part).permute(
            0, 2, 1)
        self.part_ins_att = self.avgpool(self.reweight_part).squeeze(-1)
        ins_att_weight = F.linear(output_part,
                                  self.part_ins_att)  # this is for global part-aware instance ptt which is not used in ours [NHW, C] x[C, M] = [NHW, M]
        ins_att_weight = F.softmax(ins_att_weight, dim=1)  # NHW x LT
        if self.shrink_thres > 0:
            ins_att_weight = hard_shrink_relu(ins_att_weight, lambd=self.shrink_thres)
            ins_att_weight = F.normalize(ins_att_weight, p=1, dim=1)

        ins_mem_trans = self.part_ins_att.permute(1, 0)  # Mem^T, MxC
        output_ins = F.linear(ins_att_weight, ins_mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC

        ### for semantic PTT
        # pdb.set_trace()
        self.reweight_ins = self.part_ins_att.view(self.num_cls, self.ptt_num, self.fea_dim)
        self.reweight_ins = (torch.sigmoid(self.reweight_layer_ins(self.reweight_ins)) * self.reweight_ins).permute(0,
                                                                                                                    2,
                                                                                                                    1)
        self.sem_att = self.avgpool(self.reweight_ins).squeeze(-1)
        sem_att_weight = F.linear(output_ins, self.sem_att)
        sem_att_weight = F.softmax(sem_att_weight, dim=1)

        if self.shrink_thres > 0:
            sem_att_weight = hard_shrink_relu(sem_att_weight, lambd=self.shrink_thres)
            sem_att_weight = F.normalize(sem_att_weight, p=1, dim=1)

        sem_mem_trans = self.sem_att.permute(1, 0)
        output_sem = F.linear(sem_att_weight, sem_mem_trans)

        if residual:
            output_sem += output

        # return {'output': output, 'att': att_weight}  # output, att_weight
        return {'output_sem': output_sem, 'output_part': output_part, 'output_ins': output_ins}

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule(nn.Module):
    def __init__(self, ptt_num, num_cls, part_num, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemModule, self).__init__()
        self.ptt_num = ptt_num
        self.num_cls = num_cls
        self.part_num = part_num
        ins_mem = False
        if ins_mem:
            self.mem_dim = ptt_num * num_cls * part_num  # part-level instance
        else:
            self.mem_dim = num_cls  # global semantic
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.ptt_num, self.num_cls, self.part_num, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        ###投偷懒代码###
        #self.fea_dim = input[1]
        s = input.data.shape
        l = len(s)
        ###############################
        #print("s = ",s,"l = ",l)
        #############################
        if l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
        #print("调整之后 x = ",x.shape)
        x = x.contiguous()
        x = x.view(-1, s[1])
        #print("view之后 x = ",x.shape)
        #
        y_and = self.memory(x)
        #
        y_sem = y_and['output_sem']
        y_ins = y_and['output_ins']
        y_part = y_and['output_part']
#         print("memory之后y_sem = ",y_sem.shape)
#         print("memory之后y_ins = ",y_ins.shape)
#         print("memory之后y_part = ",y_part.shape)
        if l == 4:
            y_sem = y_sem.view(s[0], s[2], s[3], s[1])
            y_sem = y_sem.permute(0, 3, 1, 2)
            y_ins = y_ins.view(s[0], s[2], s[3], s[1])
            y_ins = y_ins.permute(0, 3, 1, 2)
            y_part = y_part.view(s[0], s[2], s[3], s[1])
            y_part = y_part.permute(0, 3, 1, 2)

        else:
            print('wrong feature map size')
#         print("再次调整后：")
#         print("memory之后y_sem = ",y_sem.shape)
#         print("memory之后y_ins = ",y_ins.shape)
#         print("memory之后y_part = ",y_part.shape)
        ################添加融合#####################
        tpt = (y_sem + y_ins + y_part) / 3
        #print(y_sem[0][0],y_ins[0][0],y_part[0][0],tpt[0][0])
        #print(tpt.shape)
        return y_sem, y_ins, y_part
        #############返回融合值#########################
        #return tpt


# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output



####fea_dim等于通道数就可以###########
###输入[1,3,512,512],就会输出三个[1,3,512,512]
p = MemModule(ptt_num = 2, num_cls = 10, part_num = 5, fea_dim = 10)
image = torch.rand(1,10,512,512);
writer = SummaryWriter()
writer.add_graph(p, input_to_model=image, verbose=False)
#print(p)
y = p(image)
writer.flush()
writer.close()

#self.mem_dim self.fea_dim  最后要和 [b,a * c * d]