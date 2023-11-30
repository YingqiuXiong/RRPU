# python3.6                                
# encoding    : utf-8 -*-                            
# @author     : YingqiuXiong
# @e-mail     : 1916728303@qq.com                                    
# @file       : rrpu.py
# @Time       : 2021/11/23 10:44

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class RRPU(nn.Module):
    def __init__(self, paramsConfig):
        super(RRPU, self).__init__()
        self.paramsConfig = paramsConfig

        self.word_embs = nn.Embedding(self.paramsConfig.vocab_size, self.paramsConfig.word_dim)  # vocab_size * 300
        self.word_embs.weight.requires_grad = False

        # user representation
        self.user_net = UserNet(paramsConfig, self.word_embs)  # 用户评论表征用户
        self.u_id_embedding = nn.Embedding(self.paramsConfig.user_num, self.paramsConfig.id_emb_size)  # embedding用户的Id

        # item representation
        self.item_net = ItemNet(paramsConfig, self.word_embs)
        self.i_id_embedding = nn.Embedding(self.paramsConfig.item_num, self.paramsConfig.id_emb_size)  # embedding产品的Id

        # prior to h(MLP),p(h|user)
        self.g_theta = nn.Sequential(
            nn.Linear(self.paramsConfig.fc_dim, self.paramsConfig.fc_dim),
            nn.Tanh(),
            nn.Linear(self.paramsConfig.fc_dim, self.paramsConfig.h_dim),
            nn.Tanh()
        )
        self.f_mean_theta = nn.Linear(self.paramsConfig.h_dim, self.paramsConfig.h_dim)
        self.f_log_var_theta = nn.Linear(self.paramsConfig.h_dim, self.paramsConfig.h_dim)

        # posterior to h(MLP),q(h|user, item, label)
        input_dim_phi = self.paramsConfig.fc_dim * 2 + self.paramsConfig.label_dim
        self.g_phi = nn.Sequential(
            nn.Linear(input_dim_phi, input_dim_phi),
            nn.Tanh(),
            nn.Linear(input_dim_phi, self.paramsConfig.h_dim),
            nn.Tanh()
        )
        self.f_mean_phi = nn.Linear(self.paramsConfig.h_dim, self.paramsConfig.h_dim)
        self.f_log_var_phi = nn.Linear(self.paramsConfig.h_dim, self.paramsConfig.h_dim)

        # attention component
        self.w_h = nn.Parameter(torch.Tensor(self.paramsConfig.h_dim, self.paramsConfig.h_dim))  # [32, 32]
        self.w_aspect = nn.Parameter(torch.Tensor(self.paramsConfig.h_dim, self.paramsConfig.fc_dim))  # [32, 50]
        self.v_att = nn.Parameter(torch.Tensor(self.paramsConfig.h_dim, 1))  # [32, 1]

        self.dropout = nn.Dropout(self.paramsConfig.drop_out)
        self.item_fc_layer = nn.Linear(self.paramsConfig.fc_dim, self.paramsConfig.fc_dim)

        # Initialize all weights
        self.reset_para()

    def forward(self, datas, scores):
        """
        learn user and item representation
        :param datas: a batch of dataset
        :param scores: a batch of user-item score
        :return: user-feature and item-feature
        """
        # 数据
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas
        user_num = len(uids)
        # aa = 0
        # with open(self.paramsConfig.user2Items, "a", encoding="utf-8") as f:
        #     for i in range(user_num):
        #         line = ""
        #         user_id = int(uids[i])
        #         if user_id not in self.paramsConfig.exist_userids:
        #             self.paramsConfig.exist_userids.append(user_id)
        #         else:
        #             continue
        #         aa += 1
        #         line += str(user_id) + ": "
        #         item_ids = user_item2id[i].tolist()
        #         for item_id in item_ids:
        #             if item_id == 2421:
        #                 continue
        #             line += str(item_id) + "\t"
        #         f.write(line + "\n")
        # print(f"*****the number of users in current batch: {aa}*****")

        self.paramsConfig.batch_user_ids = uids.tolist()
        self.paramsConfig.batch_item_ids = iids.tolist()
        # 用户表征
        u_fea = self.user_net(user_reviews, iids, user_item2id)  # 只从评论中提取的特征, 每个用户表征为32维vector, [128, 32]
        u_id_emb = self.u_id_embedding(uids)  # 从id中提取的特征，建模的是用户的评分习惯
        u_feature = torch.cat([u_id_emb, u_fea], 1)  # concat review feature and id feature
        # u_feature = torch.add(u_id_emb, u_fea)
        # 产品表征
        batch_aspAttn, batch_aspRep = self.item_net(item_doc)
        batch_aspRep_att = torch.unsqueeze(batch_aspRep, 3)  # 计算attention需要， [128, 15, 50, 1]
        i_id_emb = self.i_id_embedding(iids)  # 从id中提取的特征，建模的是产品的整体评分特征
        # 根据u生成隐变量的先验参数,即p(h|u)
        pi_theta = self.g_theta(u_fea)
        mean_theta = self.f_mean_theta(pi_theta)
        log_var_theta = self.f_log_var_theta(pi_theta)
        # 训练的时候需要推断隐变量，并计算多个产品的表征，损失函数还包括kl_loss
        if self.paramsConfig.isTrain:
            # 标签
            labels = [int(score - 1) for score in scores]  # 0, 1, 2, 3, 4
            labels = torch.eye(5)[labels, :]  # 评分转换为one-hot表示, [128, 5]
            if self.paramsConfig.use_gpu:
                labels = labels.cuda()
            # inference network: 根据可观测变量，即用户，产品及评分推理隐变量h, 即生成隐变量的近似后验参数
            item_rep = torch.mean(batch_aspRep, dim=1)  # [128, 50]
            mean_phi, log_var_phi = self.encoder(x_u=u_fea, x_i=item_rep, r=labels)  # 近似后验均值和方差
            kl_loss = self.multivar_continue_KL_divergence(
                mu1=mean_phi, mu2=mean_theta, logvar_1=log_var_phi, logvar_2=log_var_theta)
            i_fea_times = []
            for i in range(self.paramsConfig.sample_times):
                h = self.sample(mean=mean_phi, log_var=log_var_phi)  # [128, 32]
                # 用推断出的h计算attention
                att_score = self.attention(h, batch_aspRep_att)  # [128, 15, 1]
                # [128, 15, 50] x [128, 15, 1]
                batch_ItemRep = torch.mul(batch_aspRep, att_score)  # [128, 15, 50]
                batch_ItemRep = torch.sum(batch_ItemRep, dim=1)  # [128, 50], 最终的从评论得到的产品表征
                batch_ItemRep = self.dropout(batch_ItemRep)
                batch_ItemRep = self.item_fc_layer(batch_ItemRep)
                i_fea = torch.cat([i_id_emb, batch_ItemRep], 1)  # [128, 2, 32]
                # i_fea = torch.add(i_id_emb, batch_ItemRep)
                i_fea_times.append(i_fea)
            return u_feature, i_fea_times, kl_loss
        # 测试的时候只需要顺着网络前向传播
        else:
            i_fea_times = []
            for i in range(self.paramsConfig.sample_times):
                h = self.sample(mean=mean_theta, log_var=log_var_theta)  # 先验中采样
                # h = mean_theta
                # 用推断出的h计算attention
                att_score = self.attention(h, batch_aspRep_att)  # [128, 5, 1]
                # [128, 5, 32] x [128, 5, 1]
                batch_ItemRep = torch.mul(batch_aspRep, att_score)  # [128, 15, 50]
                batch_ItemRep = torch.sum(batch_ItemRep, dim=1)  # [128, 50], 最终的从评论得到的产品表征
                batch_ItemRep = self.dropout(batch_ItemRep)
                batch_ItemRep = self.item_fc_layer(batch_ItemRep)
                i_feature = torch.cat([i_id_emb, batch_ItemRep], 1)
                i_fea_times.append(i_feature)
            # i_feature = torch.add(i_id_emb, batch_ItemRep)
            return u_feature, i_fea_times

    def multivar_continue_KL_divergence(self, mu1, mu2, logvar_1, logvar_2):
        mu1, mu2 = mu1.type(dtype=torch.float64), mu2.type(dtype=torch.float64)
        sigma_1 = logvar_1.exp().type(dtype=torch.float64)
        sigma_2 = logvar_2.exp().type(dtype=torch.float64)

        sigma_diag_1 = torch.diag_embed(sigma_1, offset=0, dim1=-2, dim2=-1)
        sigma_diag_2 = torch.diag_embed(sigma_2, offset=0, dim1=-2, dim2=-1)

        sigma_diag_2_inv = sigma_diag_2.inverse()

        # log(det(sigma2^T)/det(sigma1))
        term_1 = (sigma_diag_2.det() / sigma_diag_1.det()).log()

        # trace(inv(sigma2)*sigma1)
        term_2 = torch.diagonal((torch.matmul(sigma_diag_2_inv, sigma_diag_1)), dim1=-2, dim2=-1).sum(-1)

        # (mu2-m1)^T*inv(sigma2)*(mu2-mu1)
        term_3 = torch.matmul(torch.matmul((mu2 - mu1).unsqueeze(-1).transpose(2, 1), sigma_diag_2_inv),
                              (mu2 - mu1).unsqueeze(-1)).flatten()

        # dimension of embedded space (number of mus and sigmas)
        n = mu1.shape[1]

        # Calc kl divergence on entire batch
        kl = 0.5 * (term_1 - n + term_2 + term_3)
        kl_agg = torch.mean(kl)

        return kl_agg

    # 用推断出的h以及item's aspect-level representation 计算attention
    def attention(self, h, aspRep_att):
        h_att = torch.unsqueeze(h, 2)  # [128, 32, 1]
        # [128, 15, 32, 1]
        att = torch.unsqueeze(torch.matmul(self.w_h, h_att), 1) + torch.matmul(self.w_aspect, aspRep_att)
        att = torch.tanh(att)
        att = torch.matmul(torch.transpose(self.v_att, 0, 1), att)  # [128, 15, 1, 1]
        att = torch.squeeze(att)  # [128, 15]

        if len(att.size()) == 1:
            att = att.unsqueeze(0)

        att_score = F.softmax(att, dim=1)
        att_score = torch.unsqueeze(att_score, dim=2)  # [128, 15, 1]
        return att_score

    # 重参数采样：从先验或近似后验中采样隐变量h
    def sample(self, mean, log_var):
        sd = torch.exp(0.5 * log_var)  # 标准差
        if torch.cuda.is_available():
            epsilon = Variable(torch.randn(sd.size()), requires_grad=False).cuda()  # Sample from standard normal
        else:
            epsilon = Variable(torch.randn(sd.size()), requires_grad=False)
        h = mean + torch.multiply(epsilon, sd)
        return h

    # 拟合近似后验posterior，从而采样推断出隐变量h
    def encoder(self, x_u, x_i, r):
        """
        Inference Network, p(h|u, i, r)
        Construct vector representations of the observed variables:
        :param x_u: user's reviews representation
        :param x_i: item's reviews representation
        :param r: five dimension one-hot vector which represents user-item rating
        """
        cat_tensor = torch.cat((x_u, x_i, r), dim=1)  # 将三个张量按列拼接，128*(32+50+5)
        pi_phi = self.g_phi(cat_tensor)
        mean_phi = self.f_mean_phi(pi_phi)
        log_var_phi = self.f_log_var_phi(pi_phi)
        return mean_phi, log_var_phi

    def reset_para(self):
        # nn.init.uniform_(self.u_id_embedding.weight, a=-0.1, b=0.1)
        # nn.init.uniform_(self.i_id_embedding.weight, a=-0.1, b=0.1)

        for layer in [self.f_mean_theta, self.f_log_var_theta, self.f_mean_phi, self.f_log_var_phi]:
            nn.init.uniform_(layer.weight, -0.1, 0.1)
            nn.init.constant_(layer.bias, 0.1)

        for layer in [self.g_theta[0], self.g_theta[2]]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.1)

        for layer in [self.g_phi[0], self.g_phi[2]]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.1)

        nn.init.uniform_(self.w_h, -0.1, 0.1)
        nn.init.uniform_(self.w_aspect, -0.1, 0.1)
        nn.init.uniform_(self.v_att, -0.1, 0.1)

        nn.init.uniform_(self.item_fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.item_fc_layer.bias, 0.1)

        if self.paramsConfig.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.paramsConfig.w2v_path))
            if self.paramsConfig.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)


# no attention
class RRPUONE(nn.Module):
    def __init__(self, paramsConfig):
        super(RRPUONE, self).__init__()
        self.paramsConfig = paramsConfig

        self.word_embs = nn.Embedding(self.paramsConfig.vocab_size, self.paramsConfig.word_dim)  # vocab_size * 300
        self.word_embs.weight.requires_grad = False

        # user representation
        self.user_net = UserNet(paramsConfig, self.word_embs)  # 用户评论表征用户
        self.u_id_embedding = nn.Embedding(self.paramsConfig.user_num, self.paramsConfig.id_emb_size)  # embedding用户的Id

        # item representation
        self.item_net = ItemNet(paramsConfig, self.word_embs)
        self.i_id_embedding = nn.Embedding(self.paramsConfig.item_num, self.paramsConfig.id_emb_size)  # embedding产品的Id

        self.reset_para()

    def forward(self, datas):
        """
        learn user and item representation
        :param datas: a batch of dataset
        :return: user-feature and item-feature
        """
        # 数据
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas
        # 用户表征
        u_fea = self.user_net(user_reviews, iids, user_item2id)  # 只从评论中提取的特征, 每个用户表征为32维vector, [128, 32]
        u_id_emb = self.u_id_embedding(uids)  # 从id中提取的特征，建模的是用户的评分习惯
        u_feature = torch.cat([u_id_emb, u_fea], 1)  # concat review feature and id feature
        # 产品表征
        # 第一，第三种求解
        batch_aspAttn, batch_aspRep = self.item_net(item_doc)
        # 第二种求解
        # batch_aspRep = self.item_net(item_doc)
        i_id_emb = self.i_id_embedding(iids)  # 从id中提取的特征，建模的是产品的整体评分特征
        batch_aspRep = torch.mean(batch_aspRep, dim=1)  # [128, 50]
        i_feature = torch.cat([i_id_emb, batch_aspRep], 1)  # [128, 2, 32]
        return u_feature, i_feature

    def reset_para(self):
        nn.init.uniform_(self.u_id_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.i_id_embedding.weight, a=-0.1, b=0.1)

        if self.paramsConfig.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.paramsConfig.w2v_path))
            if self.paramsConfig.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)


# definite attention
class RRPUTWO(nn.Module):
    def __init__(self, paramsConfig):
        super(RRPUTWO, self).__init__()
        self.paramsConfig = paramsConfig

        self.word_embs = nn.Embedding(self.paramsConfig.vocab_size, self.paramsConfig.word_dim)  # vocab_size * 300
        self.word_embs.weight.requires_grad = False

        # user representation
        self.user_net = UserNet(paramsConfig, self.word_embs)  # 用户评论表征用户
        self.u_id_embedding = nn.Embedding(self.paramsConfig.user_num, self.paramsConfig.id_emb_size)  # embedding用户的Id


        # item representation
        self.item_net = ItemNet(paramsConfig, self.word_embs)
        self.i_id_embedding = nn.Embedding(self.paramsConfig.item_num, self.paramsConfig.id_emb_size)  # embedding产品的Id

        # prior to h(MLP),p(h|user)
        self.g_theta = nn.Sequential(
            nn.Linear(self.paramsConfig.fc_dim, self.paramsConfig.fc_dim),
            nn.Tanh(),
            nn.Linear(self.paramsConfig.fc_dim, self.paramsConfig.h_dim),
            nn.Tanh()
        )
        self.f_mean_theta = nn.Linear(self.paramsConfig.h_dim, self.paramsConfig.h_dim)

        # attention component
        self.w_h = nn.Parameter(torch.Tensor(self.paramsConfig.h_dim, self.paramsConfig.h_dim))  # [32, 32]
        self.w_aspect = nn.Parameter(torch.Tensor(self.paramsConfig.h_dim, self.paramsConfig.fc_dim))  # [32, 50]
        self.v_att = nn.Parameter(torch.Tensor(self.paramsConfig.h_dim, 1))  # [32, 1]

        self.dropout = nn.Dropout(self.paramsConfig.drop_out)
        self.item_fc_layer = nn.Linear(self.paramsConfig.fc_dim, self.paramsConfig.fc_dim)

        # Initialize all weights
        self.reset_para()

    def forward(self, datas):
        """
        learn user and item representation
        :param datas: a batch of dataset
        :param scores: a batch of user-item score
        :return: user-feature and item-feature
        """
        # 数据
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc = datas
        # 用户表征
        u_fea = self.user_net(user_reviews, iids, user_item2id)  # 只从评论中提取的特征, 每个用户表征为32维vector, [128, 32]
        u_id_emb = self.u_id_embedding(uids)  # 从id中提取的特征，建模的是用户的评分习惯
        u_feature = torch.cat([u_id_emb, u_fea], 1)  # concat review feature and id feature
        # 产品表征
        # 第一，第三种求解
        batch_aspAttn, batch_aspRep = self.item_net(item_doc)
        # 第二种求解
        # batch_aspRep = self.item_net(item_doc)
        batch_aspRep_att = torch.unsqueeze(batch_aspRep, 3)  # 计算attention需要， [128, 15, 50, 1]
        i_id_emb = self.i_id_embedding(iids)  # 从id中提取的特征，建模的是产品的整体评分特征
        # 根据u生成隐变量的先验参数,即p(h|u)
        pi_theta = self.g_theta(u_fea)
        mean_theta = self.f_mean_theta(pi_theta)
        # 用推断出的h计算attention
        att_score = self.attention(mean_theta, batch_aspRep_att)  # [128, 15, 1]
        # [128, 15, 50] x [128, 15, 1]
        batch_ItemRep = torch.mul(batch_aspRep, att_score)  # [128, 15, 50]
        batch_ItemRep = torch.sum(batch_ItemRep, dim=1)  # [128, 50], 最终的从评论得到的产品表征
        batch_ItemRep = self.dropout(batch_ItemRep)
        batch_ItemRep = self.item_fc_layer(batch_ItemRep)
        i_feature = torch.cat([i_id_emb, batch_ItemRep], 1)  # [128, 2, 32]
        return u_feature, i_feature

    # 用推断出的h以及item's aspect-level representation 计算attention
    def attention(self, h, aspRep_att):
        h_att = torch.unsqueeze(h, 2)  # [128, 32, 1]
        # [128, 15, 32, 1]
        att = torch.unsqueeze(torch.matmul(self.w_h, h_att), 1) + torch.matmul(self.w_aspect, aspRep_att)
        att = torch.tanh(att)
        att = torch.matmul(torch.transpose(self.v_att, 0, 1), att)  # [128, 15, 1, 1]
        att = torch.squeeze(att)  # [128, 15]
        if len(att.size()) == 1:
            att = att.unsqueeze(0)
        att_score = F.softmax(att, dim=1)
        att_score = torch.unsqueeze(att_score, dim=2)  # [128, 15, 1]
        return att_score

    def reset_para(self):
        nn.init.uniform_(self.u_id_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.i_id_embedding.weight, a=-0.1, b=0.1)

        for layer in [self.f_mean_theta, self.item_fc_layer]:
        # for layer in [self.item_fc_layer]:
            nn.init.uniform_(layer.weight, -0.1, 0.1)
            nn.init.constant_(layer.bias, 0.1)

        for layer in [self.g_theta[0], self.g_theta[2]]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.1)

        nn.init.uniform_(self.w_h, -0.1, 0.1)
        nn.init.uniform_(self.w_aspect, -0.1, 0.1)
        nn.init.uniform_(self.v_att, -0.1, 0.1)

        if self.paramsConfig.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.paramsConfig.w2v_path))
            if self.paramsConfig.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)


# Item ID-based User Representation Learning
class UserNet(nn.Module):
    def __init__(self, paramsConfig, word_embs):
        super(UserNet, self).__init__()
        self.paramsConfig = paramsConfig

        item_num = self.paramsConfig.item_num

        self.word_embs = word_embs

        self.i_id_embedding = nn.Embedding(item_num, 16)  # embedding产品的
        self.i_id_embedding.weight.requires_grad = True

        self.cnn = nn.Conv2d(1, paramsConfig.review_filters_num, (paramsConfig.kernel_size, paramsConfig.word_dim))

        # 从user的review中最终提取的特征维度
        self.fc_layer = nn.Linear(self.paramsConfig.review_filters_num, self.paramsConfig.fc_dim)

        self.dropout = nn.Dropout(self.paramsConfig.drop_out)
        self.reset_para()

    # 借助产品id筛选用户评论来表征用户
    def forward(self, reviews, target_item_ids, ids_list):
        # --------------- Word Embedding Layer ----------------------------------
        reviews = self.word_embs(reviews)  # (128, 13, 202, 300),每个用户13条评论，每个评论202个词
        bs, r_num, r_len, wd = reviews.size()  # batch_size, review_num, review_length, word_dimension
        reviews = reviews.view(-1, r_len, wd)  # (128*13, 202, 300), 将所有用户的评论放在一起

        # -----------cnn for all users' review--------------------
        fea = F.relu(self.cnn(reviews.unsqueeze(1))).squeeze(3)  # .permute(0, 2, 1)
        fea = F.max_pool1d(fea, fea.size(2)).squeeze(2)
        fea = fea.view(-1, r_num, fea.size(1))

        # ------------------attention--target item id embedding-------------------------------
        t_id_emb = self.i_id_embedding(target_item_ids)

        i_id_emb = self.i_id_embedding(ids_list)

        att_weight = self.attention_target_id(t_id_emb, i_id_emb)
        att_weight = torch.unsqueeze(att_weight, dim=2)

        # case study1
        # batch_size = len(target_item_ids)
        # with open(self.paramsConfig.store_case1_study, "a", encoding="utf-8") as f:
        #     for batch_num in range(batch_size):
        #         target_item_id = str(int(target_item_ids[batch_num]))
        #         f.write(self.paramsConfig.id2item[target_item_id] + "\n")
        #         reviewed_item_ids = ids_list[batch_num].tolist()
        #         attentions = att_weight[batch_num].tolist()
        #         num_items = len(reviewed_item_ids)
        #         id2attention = {}
        #         for i in range(num_items):
        #             id2attention[str(i)] = attentions[i]
        #         sorted_id2attention = sorted(id2attention.items(), key=lambda x: x[1])
        #         for (id, attn) in sorted_id2attention:
        #             f.write(str(self.paramsConfig.id2item[str(reviewed_item_ids[int(id)])]) + ": " + str(format(attn, '.4f')) + "\n")
        #         f.write("\n")
        # print("*" * 10, "one batch finished!!!", "*" * 10)

        # att_weight = self.attention_target_id(t_id_emb, i_id_emb)
        # att_weight = torch.unsqueeze(att_weight, dim=2)
        att_fea = fea * att_weight
        att_fea = att_fea.sum(dim=1)
        # mean_fea = torch.mean(fea, dim=1)  # ablation--without personalized preference
        r_fea = self.dropout(att_fea)
        return self.fc_layer(r_fea)  # [128, 32]

    # def attention_user(self, reviews_emb, items_id_emb):
    #     rs_mix = F.relu(self.review_linear(reviews_emb) + self.id_linear(F.relu(items_id_emb)))
    #     att_score = self.attention_linear(rs_mix)
    #     att_weight = F.softmax(att_score, 1)
    #     return att_weight

    def attention_target_id(self, t_id_emb, items_id_emb):
        # t_id_emb [128, 32]
        # items_id_emb [128, 13, 32]
        t_id_emb = F.normalize(t_id_emb, p=2, dim=1)
        t_id_emb = torch.unsqueeze(t_id_emb, dim=1)  # 扩充一维，[128, 1, 32]
        items_id_emb = F.normalize(items_id_emb, p=2, dim=2)
        items_id_emb = torch.transpose(items_id_emb, 1, 2)  # 转置，[128, 32, 13]
        a = torch.matmul(t_id_emb, items_id_emb)
        att_score = torch.squeeze(a, dim=1)  # [128, 14]
        att_weight = F.softmax(att_score, dim=1)  # 归一化

        return att_weight

    def reset_para(self):
        nn.init.uniform_(self.i_id_embedding.weight, a=-0.1, b=0.1)

        nn.init.xavier_normal_(self.cnn.weight)
        nn.init.constant_(self.cnn.bias, 0.1)

        nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer.bias, 0.1)


# Aspect-based Representation Learning (ARL)
class ItemNet(nn.Module):
    def __init__(self, paramsConfig, word_embs):
        super(ItemNet, self).__init__()

        self.paramsConfig = paramsConfig

        self.word_embs = word_embs

        # Aspect-Specific Projection Matrices
        self.aspProj = nn.Parameter(torch.Tensor(self.paramsConfig.num_aspects, self.paramsConfig.word_dim,
                                                 self.paramsConfig.h1), requires_grad=True)

        # 计算每个词的权重时本地窗口其实就相当于一个卷积核，卷积操作后的特征向量每个维度的值作为对应的词的权重
        self.cnn_weight = nn.Conv2d(1, 1, (paramsConfig.ctx_win_size, paramsConfig.h1), bias=False)
        self.fc_layer = nn.Linear(self.paramsConfig.h1, self.paramsConfig.fc_dim)  # 从item的review中最终提取的特征维度

        # Initialize all weights
        self.reset_para()

    '''
    [Input]		batch_docIn:	bsz x max_doc_len x word_embed_dim
    [Output]	batch_aspRep:	bsz x num_aspects x h1
    '''
    def forward(self, batch_item_doc):
        item_doc = self.word_embs(batch_item_doc)

        lst_batch_aspRep = []
        lst_batch_aspAttn = []
        # Loop over all aspects, 每个aspect有一个映射矩阵和一个embedding卷积核
        # 映射矩阵的作用是将预训练的每个词的embedding重新映射到该aspect对应的空间
        # embedding卷积核的作用是在映射矩阵上进行卷积操作，从而得到每个词对该aspect的重要性,再用softmax归一化
        # 最后，根据词对aspect的重要性，将每个词在对应aspect空间的表征聚合作为item在该aspect level上的表征
        aspect_0 = []
        aspect_1 = []
        aspect_2 = []
        aspect_3 = []
        aspect_4 = []
        batch_size = len(self.paramsConfig.batch_user_ids)
        for a in range(self.paramsConfig.num_aspects):
            # Aspect-Specific Projection of Input Word Embeddings
            batch_aspProjDoc = torch.matmul(item_doc, self.aspProj[a])  # (bsz x max_doc_len x h1), [128, 500, 10]
            # Aspect Embedding: (bsz x h1 x 1) after transposing!

            # Context-based Word Importance
            # Calculate Attention based on the word itself, and the (self.paramsConfig.ctx_win_size - 1) / 2 word(s) before & after it
            # Pad the document
            pad_size = int((self.paramsConfig.ctx_win_size - 1) / 2)
            # (0, 0, 2, 2)前两个参数对最后一个维度有效(左右分别扩充0列)，后两个参数对倒数第二维有效（上下分别扩充2行）
            batch_aspProjDoc_padded = F.pad(batch_aspProjDoc, (0, 0, pad_size, pad_size), "constant", 0)
            batch_aspAttn = self.cnn_weight(batch_aspProjDoc_padded.unsqueeze(1)).squeeze(1)
            # batch_aspAttn = F.softmax(batch_aspAttn*1e07, dim=1)
            batch_aspAttn = F.softmax(batch_aspAttn, dim=1)

            # for i in range(batch_size):
            #     wordId_list = batch_item_doc[i].tolist()
            #     aspect_word_weights = torch.squeeze(batch_aspAttn[i], dim=1).tolist()
            #     num = 0
            #     id2weight = {}
            #     for weight in aspect_word_weights:
            #         id2weight[num] = weight
            #         num += 1
            #     sorted_id2weight = sorted(id2weight.items(), key=lambda x: x[1])
            #     sorted_id2weight = sorted_id2weight[485:]
            #     word_weights = {}
            #     for (num, attn) in sorted_id2weight:
            #         word = self.paramsConfig.id2word[str(wordId_list[num])]
            #         word_weights[word] = format(attn, '.4f')
            #     if a == 0:
            #         aspect_0.append(word_weights)
            #     elif a == 1:
            #         aspect_1.append(word_weights)
            #     elif a == 2:
            #         aspect_2.append(word_weights)
            #     elif a == 3:
            #         aspect_3.append(word_weights)
            #     else:
            #         aspect_4.append(word_weights)
            #         # print(self.paramsConfig.id2word[str(wordId_list[num])], ": ", attn)

            # Weighted Sum: Broadcasted Element-wise Multiplication & Sum over Words
            # (bsz x max_doc_len x 1) and (bsz x max_doc_len x h1) -> (bsz x h1)
            batch_aspRep = batch_aspProjDoc * batch_aspAttn.expand_as(batch_aspProjDoc)
            batch_aspRep = torch.sum(batch_aspRep, dim=1)

            # Store the results (Attention & Representation) for this aspect
            lst_batch_aspRep.append(torch.unsqueeze(batch_aspRep, 1))
            # 第一、第三种求解
            lst_batch_aspAttn.append(torch.transpose(batch_aspAttn, 1, 2))

        # with open(self.paramsConfig.store_case2_study, "a", encoding="utf-8") as f:
        #     aa = 0
        #     for i in range(batch_size):
        #         user_id = self.paramsConfig.batch_user_ids[i]
        #         item_id = self.paramsConfig.batch_item_ids[i]
        #         if str(item_id) not in self.paramsConfig.printer_ids:
        #             continue
        #         aa += 1
        #         print(item_id)
        #         line = f"user_id: {user_id}, item_id: {item_id} \n"
        #         f.write(line)
        #         aspect_dict = aspect_0[i]
        #         line = "aspect 0:" + "\n"
        #         f.write(line)
        #         for word, weight in aspect_dict.items():
        #             line = word + ": " + weight + "\n"
        #             f.write(line)
        #         # aspect_dict = aspect_1[i]
        #         # line = "aspect 1:" + "\n"
        #         # f.write(line)
        #         # for word, weight in aspect_dict.items():
        #         #     line = word + ": " + weight + "\n"
        #         #     f.write(line)
        #         # aspect_dict = aspect_2[i]
        #         # line = "aspect 2:" + "\n"
        #         # f.write(line)
        #         # for word, weight in aspect_dict.items():
        #         #     line = word + ": " + weight + "\n"
        #         #     f.write(line)
        #         # aspect_dict = aspect_3[i]
        #         # line = "aspect 3:" + "\n"
        #         # f.write(line)
        #         # for word, weight in aspect_dict.items():
        #         #     line = word + ": " + weight + "\n"
        #         #     f.write(line)
        #         # aspect_dict = aspect_4[i]
        #         # line = "aspect 4:" + "\n"
        #         # f.write(line)
        #         # for word, weight in aspect_dict.items():
        #         #     line = word + ": " + weight + "\n"
        #         #     f.write(line)
        #         f.write("\n\n")
        # print(f"*****current batch {aa} finish!!!*****")

        # Reshape the Attentions & Representations
        batch_aspRep = torch.cat(lst_batch_aspRep, dim=1)
        # batch_aspRep = self.dropout(batch_aspRep)
        # batch_aspRep:		(bsz x num_aspects x id_emb_size)
        batch_aspRep = self.fc_layer(batch_aspRep)
        batch_aspAttn = torch.cat(lst_batch_aspAttn, dim=1)
        # Returns the aspect-level attention over document words, and the aspect-based representations
        return batch_aspAttn, batch_aspRep

    def reset_para(self):
        nn.init.uniform_(self.aspProj.data, a=-0.01, b=0.01)

        nn.init.uniform_(self.cnn_weight.weight, -0.1, 0.1)  # 0.7894

        nn.init.uniform_(self.fc_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.fc_layer.bias, 0.1)

    def to_var(self, x, use_gpu=True, phase="Train"):
        if use_gpu:
            x = x.cuda()
        return Variable(x, volatile=(False if phase == "Train" else True))

