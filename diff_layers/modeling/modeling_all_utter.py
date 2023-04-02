import math
import os
import warnings
import dgl
import numpy as np
import dgl.function as fn
from dgl import DGLGraph
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.autograd import Variable

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (ElectraModel, ElectraPreTrainedModel, BertModel, BertPreTrainedModel, RobertaModel,RobertaPreTrainedModel)
from transformers.configuration_electra import ElectraConfig

relation_key_pair = {'Comment': 0, 'Clarification_question': 1, 'Elaboration': 2, 'Acknowledgement': 3, 'Continuation': 4, 'Explanation': 5, 'Conditional': 6, 'QAP': 7, 'Alternation': 8, 'Q-Elab': 9, 'Result': 10, 'Background': 11, 'Narration': 12, 'Correction': 13, 'Parallel': 14, 'Contrast': 15}

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (batch_size, seq_len, all_head_size) -> (batch_size, num_attention_heads, seq_len, attention_head_size)

    def forward(self, input_ids_a, input_ids_b, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(input_ids_a)
        mixed_key_layer = self.key(input_ids_b)
        mixed_value_layer = self.value(input_ids_b)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # (batch_size * num_choice, num_attention_heads, seq_len, attention_head_size) -> (batch_size * num_choice, seq_len, num_attention_heads, attention_head_size)

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        
        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids_a + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)

class FuseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear3 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()

    def forward(self, orig, input1, input2):
        out1 = self.activation(self.linear1(torch.cat([orig, input1, orig - input1, orig * input1], dim = -1)))
        out2 = self.activation(self.linear2(torch.cat([orig, input2, orig - input2, orig * input2], dim = -1)))
        fuse_prob = self.gate(self.linear3(torch.cat([out1, out2], dim = -1)))

        return fuse_prob * input1 + (1 - fuse_prob) * input2

class RGCNLayer(nn.Module):
    def __init__(self, feat_size, num_rels, activation=None, gated = True):
        
        super(RGCNLayer, self).__init__()
        self.feat_size = feat_size
        self.num_rels = num_rels
        self.activation = activation
        self.gated = gated

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, self.feat_size))
        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,gain=nn.init.calculate_gain('relu'))
        
        if self.gated:
            self.gate_weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, 1))
            nn.init.xavier_uniform_(self.gate_weight,gain=nn.init.calculate_gain('sigmoid'))
        
    def forward(self, g):
        
        weight = self.weight
        gate_weight = self.gate_weight
        
        def message_func(edges):
            w = weight[edges.data['rel_type']]
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            msg = msg * edges.data['norm']
            
            if self.gated:
                gate_w = gate_weight[edges.data['rel_type']]
                gate = torch.bmm(edges.src['h'].unsqueeze(1), gate_w).squeeze().reshape(-1,1)
                gate = torch.sigmoid(gate)
                msg = msg * gate    
            return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class RGCNModel(nn.Module):
    def __init__(self, h_dim, num_rels, num_hidden_layers=1, gated = True):
        super(RGCNModel, self).__init__()

        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_hidden_layers = num_hidden_layers
        self.gated = gated
        
        # create rgcn layers
        self.build_model()
       
    def build_model(self):        
        self.layers = nn.ModuleList() 
        for _ in range(self.num_hidden_layers):
            rgcn_layer = RGCNLayer(self.h_dim, self.num_rels, activation=F.relu, gated = self.gated)
            self.layers.append(rgcn_layer)
    
    def forward(self, g):
        for layer in self.layers:
            layer(g)
        
        rst_hidden = []
        if isinstance(g, DGLGraph):
            rst_hidden.append(g.ndata['h'])
            # print("dglgraph")
        else:
            print("batcheddglgraph")
            for sub_g in dgl.unbatch(g):
                rst_hidden.append(sub_g.ndata['h'])
        return rst_hidden


class BertForQuestionAnswering(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, the_device='cpu', num_decoupling=1):
        super().__init__(config)
        self.the_device = the_device
        # print('the device:', self.the_device)
        self.num_labels = config.num_labels
        self.num_decoupling = num_decoupling
        # print("num_labels:",config.num_labels)
        # print("config.hidden_size:",config.hidden_size)
        self.bert = BertModel(config, add_pooling_layer=False)

        self.SASelfMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.SACrossMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.fuse2 = FuseLayer(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.relation_embeds = nn.Embedding(18, config.hidden_size) #0-15relations, 16cls, 17padding
        # self.dGCN = RGCNModel(config.hidden_size, 6, 3, True)
        # self.sGCN = RGCNModel(config.hidden_size, 3, 3, True)
        self.qa_outputs = nn.Linear(3*config.hidden_size, config.num_labels)
        self.has_ans = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        relations=None, #gcn_add
        is_impossibles=None,
        speaker_ids=None,
        cls_index=None,
        turn_ids=None,
        sep_positions=None
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        self.the_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print('the device:', self.the_device)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print('input_ids:',input_ids.size())
        # print('attention_mask:',attention_mask)
        # print('token_type_ids:',token_type_ids.size())
        speaker_ids_bert = speaker_ids
        orig_speaker_ids = speaker_ids
        turn_ids = turn_ids.unsqueeze(-1).repeat([1,1,turn_ids.size(1)])
        speaker_ids = speaker_ids.unsqueeze(-1).repeat([1,1,speaker_ids.size(1)])

        original_attention_mask = attention_mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        local_mask = torch.zeros_like(attention_mask, dtype = self.dtype)
        local_mask = local_mask.repeat((1,1,attention_mask.size(-1), 1)) #(batch_size * num_choice, 1, seq_len, seq_len)
        # global_mask = torch.zeros_like(local_mask, dtype = self.dtype)
        sa_self_mask = torch.zeros_like(local_mask, dtype = self.dtype)
        sa_cross_mask = torch.zeros_like(local_mask, dtype = self.dtype)

        last_seps = []

        for i in range(input_ids.size(0)):
            last_sep = 1

            while last_sep < len(sep_positions[i]) and sep_positions[i][last_sep] != 0: 
                last_sep += 1
            
            last_sep = last_sep - 1
            last_seps.append(last_sep)

            sa_self_mask[i, 0, speaker_ids[i] == speaker_ids[i].T] = 1.0
            sa_self_mask[i, 0, :, (sep_positions[i][last_sep] + 1):] = 0
            # print('sa_self_mask_index', speaker_ids[i]==speaker_ids[i].T)
            
            sa_cross_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)] = 1.0 - sa_self_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)]

        attention_mask = (1.0 - attention_mask) * -10000.0
        sa_self_mask = (1.0 - sa_self_mask) * -10000.0
        sa_cross_mask = (1.0 - sa_cross_mask) * -10000.0
        
        # for i in range(speaker_ids_bert.size(0)):
        #     for j in range(speaker_ids_bert.size(1)):
        #         if speaker_ids_bert[i][j]==-2 or speaker_ids_bert[i][j]==-1:
        #             speaker_ids_bert[i][j]=0
        #         else:
        #             speaker_ids_bert[i][j]+=1
        # print("speaker_ids_bert:",speaker_ids_bert)

        outputs = self.bert(
            input_ids,
            attention_mask=original_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # speaker_ids=speaker_ids_bert
        )

        sequence_output = outputs[0]
        # print("sequence_output:",sequence_output.size())
        sa_self_word_level = self.SASelfMHA[0](sequence_output, sequence_output, attention_mask = sa_self_mask)[0]
        sa_cross_word_level = self.SACrossMHA[0](sequence_output, sequence_output, attention_mask = sa_cross_mask)[0]
        for t in range(1, self.num_decoupling):
            sa_self_word_level = self.SASelfMHA[t](sa_self_word_level, sa_self_word_level, attention_mask = sa_self_mask)[0]
            sa_cross_word_level = self.SACrossMHA[t](sa_cross_word_level, sa_cross_word_level, attention_mask = sa_cross_mask)[0]
        sa_word_level = self.fuse2(sequence_output, sa_self_word_level, sa_cross_word_level)
        

        batch_size = sequence_output.size(0)
        seq_len = sequence_output.size(1)
        hidden_size = sequence_output.size(2)
        app0 = torch.ones(batch_size, seq_len, hidden_size).to(self.the_device)        
        for batchi in range(batch_size):
            # #every sample in a batch
            sep_pos = [] #=sep_positions[batchi]
            for j in range(input_ids.size(1)):
                if input_ids[batchi][j]==102:
                    sep_pos.append(j)
            # print('input_ids,',input_ids[batchi])
            # print('sep_pos,',sep_pos)
            num_utterance = len(sep_pos)-1
            #sep_embedding->utterance representation
            sep_embedding = torch.ones(num_utterance, hidden_size).to(self.the_device)
            for sep_idx in range(len(sep_pos)-1):
                sep_embedding[sep_idx] = sequence_output[batchi,sep_pos[sep_idx+1],:]#+1
            # dG = dgl.DGLGraph()
            # # print('+++++++++++++++++++++++graph:', G)
            # dG.to(self.the_device)
            # # print('+++++++++++++++-----++++graph:', G)
            relation = []
            edge_type = []    # in total six type of edges
            edges = []
            relation_batch = relations[batchi, :, :]
            # print('relation_batch:', relation_batch.size())
            for idx in range(relation_batch.size(0)):
                if relation_batch[idx][2]==-1:
                    rel_len = idx
                    break
                if relation_batch[idx][2] not in relation:
                    relation.append(relation_batch[idx][2])
            else:
                rel_len = 13
            # print('relation_batch:', relation_batch)
            # print("rel_len:", rel_len)
            # print('relation_batch:', relation_batch[0:rel_len,:])

            # dG.add_nodes(num_utterance + 1 + len(relation))       # total utterance nodes in the graph

            # # constructing graph by adding edge one by one
            # for item in relation_batch[0:rel_len,:].cpu().numpy().tolist():
            #     # add default_in and default_out edges
            #     dG.add_edges(item[0], (relation.index(item[2])+num_utterance+1))
            #     edge_type.append(0)
            #     edges.append([item[0], (relation.index(item[2])+num_utterance+1)])
            #     dG.add_edges((relation.index(item[2])+num_utterance+1), item[1])
            #     edge_type.append(1)
            #     edges.append([(relation.index(item[2])+num_utterance+1), item[1]])
                
            #     # add reverse_out and reverse_in edges
            #     dG.add_edges((relation.index(item[2])+num_utterance+1), item[0])
            #     edge_type.append(2)
            #     edges.append([(relation.index(item[2])+num_utterance+1), item[0]])
            #     dG.add_edges(item[1], (relation.index(item[2])+num_utterance+1))
            #     edge_type.append(3)
            #     edges.append([item[1], (relation.index(item[2])+num_utterance+1)])
                
            # # add self edges
            # for x in range(num_utterance + 1 + len(relation)):
            #     dG.add_edges(x,x)
            #     edge_type.append(4)
            #     edges.append([x,x])

            # # add global edges
            # for x in range(num_utterance + 1 + len(relation)):
            #     if x != num_utterance:
            #         dG.add_edges(num_utterance, x)
            #         edge_type.append(5)
            #         edges.append([num_utterance, x])

            # # add node feature
            # for i in range(num_utterance + 1 + len(relation)):
            #     if i < num_utterance:
            #         # print(self.relation_embeds(Variable(torch.LongTensor([16,]))))
            #         # print(sep_embedding[i].unsqueeze(0).size())
            #         dG.nodes[[i]].data['h'] = sep_embedding[i].unsqueeze(0) #sep embedding
            #     elif i == num_utterance:
            #         dG.nodes[[i]].data['h'] = self.relation_embeds(Variable(torch.LongTensor([16,]).to(self.the_device)))
            #     else:
            #         index_relation = relation[i-num_utterance-1]
            #         dG.nodes[[i]].data['h'] = self.relation_embeds(Variable(torch.LongTensor([index_relation,]).to(self.the_device)))

            # edge_norm = []
            # # print('edges:', edges)
            # for e1, e2 in edges:
            #     # print('e1,e2', e1, e2)
            #     if e1 == e2:
            #         edge_norm.append(1)
            #         # print('e1=e2')
            #     else:
            #         edge_norm.append(1/(dG.in_degrees(e2)-1))
            #         # print('e1!=e2', 1/(G.in_degrees(e2)-1), G.in_degrees(e2))
                    
            # edge_type = torch.from_numpy(np.array(edge_type)).to(self.the_device)
            # # print(edge_norm)
            # # print(np.array(edge_norm))
            # # print(torch.from_numpy(np.array(edge_norm)))
            # edge_norm = torch.from_numpy(np.array(edge_norm)).unsqueeze(1).float().to(self.the_device)
            # dG.edata.update({'rel_type': edge_type,})
            # dG.edata.update({'norm': edge_norm})
            # X1 = self.dGCN(dG)[0]   # [bz, hdim]
            # print('gcn output:', X.size(), X)
            #make a appended matrix    
            X1 = torch.ones(num_utterance + 1 + len(relation), hidden_size)
            for i in range(num_utterance + 1 + len(relation)):
                if i < num_utterance:
                    # print(self.relation_embeds(Variable(torch.LongTensor([16,]))))
                    # print(sep_embedding[i].unsqueeze(0).size())
                    X1[i] = sep_embedding[i].unsqueeze(0) #sep embedding
                elif i == num_utterance:
                    X1[i] = self.relation_embeds(Variable(torch.LongTensor([16,]).to(self.the_device)))
                else:
                    index_relation = relation[i-num_utterance-1]
                    X1[i] = self.relation_embeds(Variable(torch.LongTensor([index_relation,]).to(self.the_device)))

            app = torch.ones(seq_len, hidden_size)
            app[0] = X1[num_utterance]
            # print("0, X[num_utterance]", app[0])
            for i in range(0, len(sep_pos)):
                if i==0:
                    for j in range(sep_pos[i]):
                        app[j+1] = self.relation_embeds(Variable(torch.LongTensor([17,])).to(self.the_device))
                        # print("j+1,padding",j+1, app[j+1])
                else:
                    for j in range(sep_pos[i-1], sep_pos[i]):
                        app[j+1] = X1[i-1]
                        # print("j+1,X[i]",j+1, X[i-1])
            for i in range(seq_len-sep_pos[-1]-1):
                app[i+sep_pos[-1]+1] = self.relation_embeds(Variable(torch.LongTensor([17,])).to(self.the_device))
            #     print("i,padding",i+sep_pos[-1]+1,app[i+sep_pos[-1]+1])
            # print("app", app)
            app0[batchi,:,:] = app[:,:]
        #cat the matrix and sequeence_output
        # # print('app0', app0)
        # with_gcn = torch.cat((sequence_output, app0), 2)
        # print('with_gcn:', with_gcn.size())

        # app01 = torch.ones(batch_size, seq_len, hidden_size).to(self.the_device)        
        # for batchi in range(batch_size):
        #     #every sample in a batch
        #     speaker_dic={} #utter->speaker
        #     sep_pos = []#sep_position
        #     for j in range(input_ids.size(1)):
        #         if input_ids[batchi][j]==102:
        #             sep_pos.append(j)
        #     num_utterance = len(sep_pos)-1
        #     for id in range(1,len(sep_pos)):
        #         speaker_dic[id-1]=orig_speaker_ids[batchi][sep_pos[id]]
        #     # print('speaker_dic:',speaker_dic)
        #     sep_embedding = torch.ones(num_utterance, hidden_size).to(self.the_device)
        #     for sep_idx in range(len(sep_pos)-1):
        #         sep_embedding[sep_idx] = sequence_output[batchi,sep_pos[sep_idx+1],:]#+1
        #     sG = dgl.DGLGraph()
        #     # print('+++++++++++++++++++++++graph:', G)
        #     sG.to(self.the_device)
        #     # print('+++++++++++++++-----++++graph:', G)
        #     relation = []
        #     edge_type = []    # in total six type of edges
        #     edges = []
           
        #     sG.add_nodes(num_utterance + 1) 
            
        #     for item in range(num_utterance):
        #         for item1 in range(item+1,num_utterance):
        #             # print('num_utter', num_utterance)
        #             # print("speaker_item, speaker_item1", speaker_dic[item],speaker_dic[item1])
        #             if speaker_dic[item]==speaker_dic[item1] is True:
        #                 sG.add_edges(item, item1)
        #                 edges.append([item, item1])
        #                 edge_type.append(0)
        #                 sG.add_edges(item1, item)
        #                 edges.append([item1, item])
        #                 edge_type.append(0)
        #     # add self edges
        #     for x in range(num_utterance + 1):
        #         sG.add_edges(x,x)
        #         edge_type.append(1)
        #         edges.append([x,x])

        #     # add global edges
        #     for x in range(num_utterance):
        #         sG.add_edges(num_utterance, x)
        #         edge_type.append(2)
        #         edges.append([num_utterance, x])

        #     # add node feature
        #     for i in range(num_utterance + 1):
        #         if i < num_utterance:
        #             # print(self.relation_embeds(Variable(torch.LongTensor([16,]))))
        #             # print(sep_embedding[i].unsqueeze(0).size())
        #             sG.nodes[[i]].data['h'] = sep_embedding[i].unsqueeze(0) #sep embedding
        #         else:           # i == num_utterance:
        #             sG.nodes[[i]].data['h'] = self.relation_embeds(Variable(torch.LongTensor([16,]).to(self.the_device)))#16
        #         # else:
        #         #     index_relation = relation[i-num_utterance-1]
        #         #     G.nodes[[i]].data['h'] = self.relation_embeds(Variable(torch.LongTensor([index_relation,]).to(self.the_device)))

        #     edge_norm = []
        #     # print('edges:', edges)
        #     for e1, e2 in edges:
        #         # print('e1,e2', e1, e2)
        #         if e1 == e2:
        #             edge_norm.append(1)
        #             # print('e1=e2')
        #         else:
        #             edge_norm.append(1/(sG.in_degrees(e2)-1))
        #             # print('e1!=e2', 1/(G.in_degrees(e2)-1), G.in_degrees(e2))
                    
        #     edge_type = torch.from_numpy(np.array(edge_type)).to(self.the_device)
        #     # print(edge_norm)
        #     # print(np.array(edge_norm))
        #     # print(torch.from_numpy(np.array(edge_norm)))
        #     edge_norm = torch.from_numpy(np.array(edge_norm)).unsqueeze(1).float().to(self.the_device)
        #     sG.edata.update({'rel_type': edge_type,})
        #     sG.edata.update({'norm': edge_norm})
        #     X2 = self.sGCN(sG)[0]   # [bz, hdim]
        #     # print('gcn output:', X.size(), X)
        #     #make a appended matrix    
                
        #     app1 = torch.ones(seq_len, hidden_size)
        #     app1[0] = X2[num_utterance]
        #     # print("0, X[num_utterance]", app[0])
        #     for i in range(0, len(sep_pos)):
        #         if i==0:
        #             for j in range(sep_pos[i]):
        #                 app1[j+1] = self.relation_embeds(Variable(torch.LongTensor([17,])).to(self.the_device))#17
        #                 # print("j+1,padding",j+1, app[j+1])
        #         else:
        #             for j in range(sep_pos[i-1], sep_pos[i]):
        #                 app1[j+1] = X2[i-1]
        #                 # print("j+1,X[i]",j+1, X[i-1])
        #     for i in range(seq_len-sep_pos[-1]-1):
        #         app1[i+sep_pos[-1]+1] = self.relation_embeds(Variable(torch.LongTensor([17,])).to(self.the_device))#17
        #     #     print("i,padding",i+sep_pos[-1]+1,app[i+sep_pos[-1]+1])
        #     # print("app", app)
        #     app01[batchi,:,:] = app1[:,:]
        #cat the matrix and sequeence_output
        with_gcn = torch.cat((sa_word_level,app0,sequence_output), 2)
        with_gcn = self.dropout(with_gcn)

        logits = self.qa_outputs(with_gcn)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        first_word = sequence_output[:, 0, :]
        has_log = self.has_ans(first_word)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(is_impossibles.size()) > 1:
                is_impossibles = is_impossibles.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_impossibles.clamp_(0, ignored_index)
        
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            choice_loss = loss_fct(has_log, is_impossibles)
            total_loss = (start_loss + end_loss + choice_loss) / 3

        if not return_dict:
            output = (start_logits, end_logits, has_log) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            has_log=has_log,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ElectraForQuestionAnswering(ElectraPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, the_device='cpu', num_decoupling=1):
        super().__init__(config)
        self.the_device = the_device
        # print('the device:', self.the_device)
        self.num_labels = config.num_labels
        self.num_decoupling = num_decoupling
        # print("num_labels:",config.num_labels)
        # print("config.hidden_size:",config.hidden_size)
        self.electra = ElectraModel(config)

        self.SASelfMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.SACrossMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.fuse2 = FuseLayer(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.relation_embeds = nn.Embedding(18, config.hidden_size) #0-15relations, 16cls, 17padding
        self.dGCN = RGCNModel(config.hidden_size, 6, 1, True)
        self.sGCN = RGCNModel(config.hidden_size, 3, 1, True)
        self.qa_outputs = nn.Linear(4*config.hidden_size, config.num_labels)
        self.has_ans = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        relations=None, #gcn_add
        is_impossibles=None,
        speaker_ids=None,
        cls_index=None,
        turn_ids=None,
        sep_positions=None
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        self.the_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print('the device:', self.the_device)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print('input_ids:',input_ids.size())
        # print('attention_mask:',attention_mask)
        # print('token_type_ids:',token_type_ids.size())
        speaker_ids_bert = speaker_ids
        orig_speaker_ids = speaker_ids
        turn_ids = turn_ids.unsqueeze(-1).repeat([1,1,turn_ids.size(1)])
        speaker_ids = speaker_ids.unsqueeze(-1).repeat([1,1,speaker_ids.size(1)])

        original_attention_mask = attention_mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        local_mask = torch.zeros_like(attention_mask, dtype = self.dtype)
        local_mask = local_mask.repeat((1,1,attention_mask.size(-1), 1)) #(batch_size * num_choice, 1, seq_len, seq_len)
        # global_mask = torch.zeros_like(local_mask, dtype = self.dtype)
        sa_self_mask = torch.zeros_like(local_mask, dtype = self.dtype)
        sa_cross_mask = torch.zeros_like(local_mask, dtype = self.dtype)

        last_seps = []

        for i in range(input_ids.size(0)):
            last_sep = 1

            while last_sep < len(sep_positions[i]) and sep_positions[i][last_sep] != 0: 
                last_sep += 1
            
            last_sep = last_sep - 1
            last_seps.append(last_sep)

            sa_self_mask[i, 0, speaker_ids[i] == speaker_ids[i].T] = 1.0
            sa_self_mask[i, 0, :, (sep_positions[i][last_sep] + 1):] = 0
            # print('sa_self_mask_index', speaker_ids[i]==speaker_ids[i].T)
            
            sa_cross_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)] = 1.0 - sa_self_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)]

        attention_mask = (1.0 - attention_mask) * -10000.0
        sa_self_mask = (1.0 - sa_self_mask) * -10000.0
        sa_cross_mask = (1.0 - sa_cross_mask) * -10000.0
        
        # for i in range(speaker_ids_bert.size(0)):
        #     for j in range(speaker_ids_bert.size(1)):
        #         if speaker_ids_bert[i][j]==-2 or speaker_ids_bert[i][j]==-1:
        #             speaker_ids_bert[i][j]=0
        #         else:
        #             speaker_ids_bert[i][j]+=1
        # print("speaker_ids_bert:",speaker_ids_bert)

        outputs = self.electra(
            input_ids,
            attention_mask=original_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # speaker_ids=speaker_ids_bert
        )

        sequence_output = outputs[0]
        # print("sequence_output:",sequence_output.size())
        sa_self_word_level = self.SASelfMHA[0](sequence_output, sequence_output, attention_mask = sa_self_mask)[0]
        sa_cross_word_level = self.SACrossMHA[0](sequence_output, sequence_output, attention_mask = sa_cross_mask)[0]
        for t in range(1, self.num_decoupling):
            sa_self_word_level = self.SASelfMHA[t](sa_self_word_level, sa_self_word_level, attention_mask = sa_self_mask)[0]
            sa_cross_word_level = self.SACrossMHA[t](sa_cross_word_level, sa_cross_word_level, attention_mask = sa_cross_mask)[0]
        sa_word_level = self.fuse2(sequence_output, sa_self_word_level, sa_cross_word_level)
        

        batch_size = sequence_output.size(0)
        seq_len = sequence_output.size(1)
        hidden_size = sequence_output.size(2)
        app0 = torch.ones(batch_size, seq_len, hidden_size).to(self.the_device)        
        for batchi in range(batch_size):
            #every sample in a batch
            sep_pos = [] #=sep_positions[batchi]
            for j in range(input_ids.size(1)):
                if input_ids[batchi][j]==102:
                    sep_pos.append(j)
            # print('input_ids,',input_ids[batchi])
            # print('sep_pos,',sep_pos)
            num_utterance = len(sep_pos)-1
            #sep_embedding->utterance representation
            sep_embedding = torch.ones(num_utterance, hidden_size).to(self.the_device)
            for sep_idx in range(len(sep_pos)-1):
                sep_embedding[sep_idx] = sequence_output[batchi,sep_pos[sep_idx+1],:]#+1
            dG = dgl.DGLGraph()
            # print('+++++++++++++++++++++++graph:', G)
            dG.to(self.the_device)
            # print('+++++++++++++++-----++++graph:', G)
            relation = []
            edge_type = []    # in total six type of edges
            edges = []
            relation_batch = relations[batchi, :, :]
            # print('relation_batch:', relation_batch.size())
            for idx in range(relation_batch.size(0)):
                if relation_batch[idx][2]==-1:
                    rel_len = idx
                    break
                if relation_batch[idx][2] not in relation:
                    relation.append(relation_batch[idx][2])
            else:
                rel_len = 13
            # print('relation_batch:', relation_batch)
            # print("rel_len:", rel_len)
            # print('relation_batch:', relation_batch[0:rel_len,:])

            dG.add_nodes(num_utterance + 1 + len(relation))       # total utterance nodes in the graph

            # constructing graph by adding edge one by one
            for item in relation_batch[0:rel_len,:].cpu().numpy().tolist():
                # add default_in and default_out edges
                dG.add_edges(item[0], (relation.index(item[2])+num_utterance+1))
                edge_type.append(0)
                edges.append([item[0], (relation.index(item[2])+num_utterance+1)])
                dG.add_edges((relation.index(item[2])+num_utterance+1), item[1])
                edge_type.append(1)
                edges.append([(relation.index(item[2])+num_utterance+1), item[1]])
                
                # add reverse_out and reverse_in edges
                dG.add_edges((relation.index(item[2])+num_utterance+1), item[0])
                edge_type.append(2)
                edges.append([(relation.index(item[2])+num_utterance+1), item[0]])
                dG.add_edges(item[1], (relation.index(item[2])+num_utterance+1))
                edge_type.append(3)
                edges.append([item[1], (relation.index(item[2])+num_utterance+1)])
                
            # add self edges
            for x in range(num_utterance + 1 + len(relation)):
                dG.add_edges(x,x)
                edge_type.append(4)
                edges.append([x,x])

            # add global edges
            for x in range(num_utterance + 1 + len(relation)):
                if x != num_utterance:
                    dG.add_edges(num_utterance, x)
                    edge_type.append(5)
                    edges.append([num_utterance, x])

            # add node feature
            for i in range(num_utterance + 1 + len(relation)):
                if i < num_utterance:
                    # print(self.relation_embeds(Variable(torch.LongTensor([16,]))))
                    # print(sep_embedding[i].unsqueeze(0).size())
                    dG.nodes[[i]].data['h'] = sep_embedding[i].unsqueeze(0) #sep embedding
                elif i == num_utterance:
                    dG.nodes[[i]].data['h'] = self.relation_embeds(Variable(torch.LongTensor([16,]).to(self.the_device)))
                else:
                    index_relation = relation[i-num_utterance-1]
                    dG.nodes[[i]].data['h'] = self.relation_embeds(Variable(torch.LongTensor([index_relation,]).to(self.the_device)))

            edge_norm = []
            # print('edges:', edges)
            for e1, e2 in edges:
                # print('e1,e2', e1, e2)
                if e1 == e2:
                    edge_norm.append(1)
                    # print('e1=e2')
                else:
                    edge_norm.append(1/(dG.in_degrees(e2)-1))
                    # print('e1!=e2', 1/(G.in_degrees(e2)-1), G.in_degrees(e2))
                    
            edge_type = torch.from_numpy(np.array(edge_type)).to(self.the_device)
            # print(edge_norm)
            # print(np.array(edge_norm))
            # print(torch.from_numpy(np.array(edge_norm)))
            edge_norm = torch.from_numpy(np.array(edge_norm)).unsqueeze(1).float().to(self.the_device)
            dG.edata.update({'rel_type': edge_type,})
            dG.edata.update({'norm': edge_norm})
            X1 = self.dGCN(dG)[0]   # [bz, hdim]
            # print('gcn output:', X.size(), X)
            #make a appended matrix    
                
            app = torch.ones(seq_len, hidden_size)
            app[0] = X1[num_utterance]
            # print("0, X[num_utterance]", app[0])
            for i in range(0, len(sep_pos)):
                if i==0:
                    for j in range(sep_pos[i]):
                        app[j+1] = self.relation_embeds(Variable(torch.LongTensor([17,])).to(self.the_device))
                        # print("j+1,padding",j+1, app[j+1])
                else:
                    for j in range(sep_pos[i-1], sep_pos[i]):
                        app[j+1] = X1[i-1]
                        # print("j+1,X[i]",j+1, X[i-1])
            for i in range(seq_len-sep_pos[-1]-1):
                app[i+sep_pos[-1]+1] = self.relation_embeds(Variable(torch.LongTensor([17,])).to(self.the_device))
            #     print("i,padding",i+sep_pos[-1]+1,app[i+sep_pos[-1]+1])
            # print("app", app)
            app0[batchi,:,:] = app[:,:]
        #cat the matrix and sequeence_output
        # # print('app0', app0)
        # with_gcn = torch.cat((sequence_output, app0), 2)
        # print('with_gcn:', with_gcn.size())

        app01 = torch.ones(batch_size, seq_len, hidden_size).to(self.the_device)        
        for batchi in range(batch_size):
            #every sample in a batch
            speaker_dic={} #utter->speaker
            sep_pos = []#sep_position
            for j in range(input_ids.size(1)):
                if input_ids[batchi][j]==102:
                    sep_pos.append(j)
            num_utterance = len(sep_pos)-1
            for id in range(1,len(sep_pos)):
                speaker_dic[id-1]=orig_speaker_ids[batchi][sep_pos[id]]
            # print('speaker_dic:',speaker_dic)
            sep_embedding = torch.ones(num_utterance, hidden_size).to(self.the_device)
            for sep_idx in range(len(sep_pos)-1):
                sep_embedding[sep_idx] = sequence_output[batchi,sep_pos[sep_idx+1],:]#+1
            sG = dgl.DGLGraph()
            # print('+++++++++++++++++++++++graph:', G)
            sG.to(self.the_device)
            # print('+++++++++++++++-----++++graph:', G)
            relation = []
            edge_type = []    # in total six type of edges
            edges = []
           
            sG.add_nodes(num_utterance + 1) 
            
            for item in range(num_utterance):
                for item1 in range(item+1,num_utterance):
                    # print('num_utter', num_utterance)
                    # print("speaker_item, speaker_item1", speaker_dic[item],speaker_dic[item1])
                    if speaker_dic[item]==speaker_dic[item1] is True:
                        sG.add_edges(item, item1)
                        edges.append([item, item1])
                        edge_type.append(0)
                        sG.add_edges(item1, item)
                        edges.append([item1, item])
                        edge_type.append(0)
            # add self edges
            for x in range(num_utterance + 1):
                sG.add_edges(x,x)
                edge_type.append(1)
                edges.append([x,x])

            # add global edges
            for x in range(num_utterance):
                sG.add_edges(num_utterance, x)
                edge_type.append(2)
                edges.append([num_utterance, x])

            # add node feature
            for i in range(num_utterance + 1):
                if i < num_utterance:
                    # print(self.relation_embeds(Variable(torch.LongTensor([16,]))))
                    # print(sep_embedding[i].unsqueeze(0).size())
                    sG.nodes[[i]].data['h'] = sep_embedding[i].unsqueeze(0) #sep embedding
                else:           # i == num_utterance:
                    sG.nodes[[i]].data['h'] = self.relation_embeds(Variable(torch.LongTensor([16,]).to(self.the_device)))#16
                # else:
                #     index_relation = relation[i-num_utterance-1]
                #     G.nodes[[i]].data['h'] = self.relation_embeds(Variable(torch.LongTensor([index_relation,]).to(self.the_device)))

            edge_norm = []
            # print('edges:', edges)
            for e1, e2 in edges:
                # print('e1,e2', e1, e2)
                if e1 == e2:
                    edge_norm.append(1)
                    # print('e1=e2')
                else:
                    edge_norm.append(1/(sG.in_degrees(e2)-1))
                    # print('e1!=e2', 1/(G.in_degrees(e2)-1), G.in_degrees(e2))
                    
            edge_type = torch.from_numpy(np.array(edge_type)).to(self.the_device)
            # print(edge_norm)
            # print(np.array(edge_norm))
            # print(torch.from_numpy(np.array(edge_norm)))
            edge_norm = torch.from_numpy(np.array(edge_norm)).unsqueeze(1).float().to(self.the_device)
            sG.edata.update({'rel_type': edge_type,})
            sG.edata.update({'norm': edge_norm})
            X2 = self.sGCN(sG)[0]   # [bz, hdim]
            # print('gcn output:', X.size(), X)
            #make a appended matrix    
                
            app1 = torch.ones(seq_len, hidden_size)
            app1[0] = X2[num_utterance]
            # print("0, X[num_utterance]", app[0])
            for i in range(0, len(sep_pos)):
                if i==0:
                    for j in range(sep_pos[i]):
                        app[j+1] = self.relation_embeds(Variable(torch.LongTensor([17,])).to(self.the_device))#17
                        # print("j+1,padding",j+1, app[j+1])
                else:
                    for j in range(sep_pos[i-1], sep_pos[i]):
                        app[j+1] = X2[i-1]
                        # print("j+1,X[i]",j+1, X[i-1])
            for i in range(seq_len-sep_pos[-1]-1):
                app[i+sep_pos[-1]+1] = self.relation_embeds(Variable(torch.LongTensor([17,])).to(self.the_device))#17
            #     print("i,padding",i+sep_pos[-1]+1,app[i+sep_pos[-1]+1])
            # print("app", app)
            app01[batchi,:,:] = app1[:,:]
        #cat the matrix and sequeence_output
        
        with_gcn = torch.cat((sa_word_level,app0,app01,sequence_output), 2)
        with_gcn = self.dropout(with_gcn)

        logits = self.qa_outputs(with_gcn)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        first_word = sequence_output[:, 0, :]
        has_log = self.has_ans(first_word)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(is_impossibles.size()) > 1:
                is_impossibles = is_impossibles.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_impossibles.clamp_(0, ignored_index)
        
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            choice_loss = loss_fct(has_log, is_impossibles)
            total_loss = (start_loss + end_loss + choice_loss) / 3

        if not return_dict:
            output = (start_logits, end_logits, has_log) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            has_log=has_log,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



