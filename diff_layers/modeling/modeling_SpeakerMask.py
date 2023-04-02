import math
import os
import warnings

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (ElectraModel, ElectraPreTrainedModel, BertModel, BertPreTrainedModel, RobertaModel,RobertaPreTrainedModel)
from transformers.configuration_electra import ElectraConfig

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



# @add_start_docstrings(
#     """
#     Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
#     layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
#     """,
#     BERT_START_DOCSTRING,
# )
class BertForSpeakerMask(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, num_decoupling=1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_decoupling = num_decoupling
        # print("num_labels:",config.num_labels)
        # print("config.hidden_size:",config.hidden_size)
        self.bert = BertModel(config, add_pooling_layer=False)

        self.localMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.globalMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.SASelfMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.SACrossMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])

        self.fuse1 = FuseLayer(config)
        self.fuse2 = FuseLayer(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.has_ans = nn.Linear(config.hidden_size, 2)

        self.qa_outputs = nn.Linear(2*config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="bert-base-uncased",
    #     output_type=QuestionAnsweringModelOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
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
        cls_index=None,
        speaker_ids=None,
        turn_ids=None,
        sep_positions=None,
        is_impossibles=None

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print('input_ids:',input_ids.size())
        # print('attention_mask:',attention_mask)
        # print('token_type_ids:',token_type_ids.size())
        speaker_ids_bert = speaker_ids
        turn_ids = turn_ids.unsqueeze(-1).repeat([1,1,turn_ids.size(1)])
        speaker_ids = speaker_ids.unsqueeze(-1).repeat([1,1,speaker_ids.size(1)])

        original_attention_mask = attention_mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        local_mask = torch.zeros_like(attention_mask, dtype = self.dtype)
        local_mask = local_mask.repeat((1,1,attention_mask.size(-1), 1)) #(batch_size * num_choice, 1, seq_len, seq_len)
        global_mask = torch.zeros_like(local_mask, dtype = self.dtype)
        sa_self_mask = torch.zeros_like(local_mask, dtype = self.dtype)
        sa_cross_mask = torch.zeros_like(local_mask, dtype = self.dtype)

        last_seps = []

        for i in range(input_ids.size(0)):
            last_sep = 1

            while last_sep < len(sep_positions[i]) and sep_positions[i][last_sep] != 0: 
                last_sep += 1
            
            last_sep = last_sep - 1
            last_seps.append(last_sep)
            # print('last_sep:',last_sep,sep_position[i][last_sep]) 

            # print('turn_ids:',turn_ids)
            # print('speaker_ids:',speaker_ids)

            local_mask[i, 0, turn_ids[i] == turn_ids[i].T] = 1.0
            local_mask[i, 0, :, (sep_positions[i][last_sep] + 1):] = 0
            # print('local_mask_index:',turn_ids[i] == turn_ids[i].T)

            sa_self_mask[i, 0, speaker_ids[i] == speaker_ids[i].T] = 1.0
            sa_self_mask[i, 0, :, (sep_positions[i][last_sep] + 1):] = 0
            # print('sa_self_mask_index', speaker_ids[i]==speaker_ids[i].T)
            
            global_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)] = 1.0 - local_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)]
            sa_cross_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)] = 1.0 - sa_self_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)]

        attention_mask = (1.0 - attention_mask) * -10000.0
        local_mask = (1.0 - local_mask) * -10000.0
        global_mask = (1.0 - global_mask) * -10000.0
        sa_self_mask = (1.0 - sa_self_mask) * -10000.0
        sa_cross_mask = (1.0 - sa_cross_mask) * -10000.0
        
        for i in range(speaker_ids_bert.size(0)):
            for j in range(speaker_ids_bert.size(1)):
                if speaker_ids_bert[i][j]==-2 or speaker_ids_bert[i][j]==-1:
                    speaker_ids_bert[i][j]=0
                else:
                    speaker_ids_bert[i][j]+=1
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
            speaker_ids=speaker_ids_bert
        )

        sequence_output = outputs[0]
        # print("sequence_output:",sequence_output.size())
        
        local_word_level = self.localMHA[0](sequence_output, sequence_output, attention_mask = local_mask)[0]
        global_word_level = self.globalMHA[0](sequence_output, sequence_output, attention_mask = global_mask)[0]
        sa_self_word_level = self.SASelfMHA[0](sequence_output, sequence_output, attention_mask = sa_self_mask)[0]
        sa_cross_word_level = self.SACrossMHA[0](sequence_output, sequence_output, attention_mask = sa_cross_mask)[0]

        for t in range(1, self.num_decoupling):
            local_word_level = self.localMHA[t](local_word_level, local_word_level, attention_mask = local_mask)[0]
            global_word_level = self.globalMHA[t](global_word_level, global_word_level, attention_mask = global_mask)[0]
            sa_self_word_level = self.SASelfMHA[t](sa_self_word_level, sa_self_word_level, attention_mask = sa_self_mask)[0]
            sa_cross_word_level = self.SACrossMHA[t](sa_cross_word_level, sa_cross_word_level, attention_mask = sa_cross_mask)[0]

        # print('local_word_level:', local_word_level.size())
        # print('global_word_level:',global_word_level.size())
        # print('sa_self_word_level:',sa_self_word_level.size())
        # print('sa_cross_word_level:',sa_cross_word_level.size())

        context_word_level = self.fuse1(sequence_output, local_word_level, global_word_level)
        sa_word_level = self.fuse2(sequence_output, sa_self_word_level, sa_cross_word_level)
        # print('context_word_level:',context_word_level.size())
        # print('sa_word_level:',sa_word_level.size())

        word_level = torch.cat((context_word_level, sa_word_level), 2)
        word_level = self.dropout(word_level) #12.1 add a dropout
        # print('word_level:',word_level.size())


        logits = self.qa_outputs(word_level)
        # print("logits:",logits.size())
        start_logits, end_logits = logits.split(1, dim=-1)
        # print("start_logits:",start_logits.size())
        # print("end_logits:",end_logits.size())
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # print("start_logits:",start_logits.size())
        # print("end_logits:",end_logits.size())

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
            # print("start_logits:",start_logits.size())
            # print("end_logits:",end_logits.size())
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            # print('ignore_index:',ignored_index)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_impossibles.clamp_(0, ignored_index)
            # print("start_logits:",start_logits.size())
            # print("end_logits:",end_logits.size())
            # print('start_positions and end_positions:',start_positions, end_positions)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # print('end_log', end_logits.size(), end_positions)
            # print('has_log', has_log.size(), is_impossibles.size())
            # print('is_impossibles:',is_impossibles)
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


# @add_start_docstrings(
#     """
#     ELECTRA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
#     layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
#     """,
#     ELECTRA_START_DOCSTRING,
# )
class ElectraForSpeakerMask(ElectraPreTrainedModel):
    config_class = ElectraConfig
    base_model_prefix = "electra"

    def __init__(self, config,  num_decoupling=1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_decoupling = num_decoupling
        self.electra = ElectraModel(config)
        
        self.localMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.globalMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.SASelfMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.SACrossMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])

        self.fuse1 = FuseLayer(config)
        self.fuse2 = FuseLayer(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.has_ans = nn.Linear(config.hidden_size, 2)

        self.qa_outputs = nn.Linear(2*config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="google/electra-small-discriminator",
    #     output_type=QuestionAnsweringModelOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
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
        cls_index=None,
        speaker_ids=None,
        turn_ids=None,
        sep_positions=None,
        is_impossibles=None
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        speaker_ids_bert = speaker_ids
        turn_ids = turn_ids.unsqueeze(-1).repeat([1,1,turn_ids.size(1)])
        speaker_ids = speaker_ids.unsqueeze(-1).repeat([1,1,speaker_ids.size(1)])

        original_attention_mask = attention_mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        local_mask = torch.zeros_like(attention_mask, dtype = self.dtype)
        local_mask = local_mask.repeat((1,1,attention_mask.size(-1), 1)) #(batch_size * num_choice, 1, seq_len, seq_len)
        global_mask = torch.zeros_like(local_mask, dtype = self.dtype)
        sa_self_mask = torch.zeros_like(local_mask, dtype = self.dtype)
        sa_cross_mask = torch.zeros_like(local_mask, dtype = self.dtype)

        last_seps = []

        for i in range(input_ids.size(0)):
            last_sep = 1

            while last_sep < len(sep_positions[i]) and sep_positions[i][last_sep] != 0: 
                last_sep += 1
            
            last_sep = last_sep - 1
            last_seps.append(last_sep)
            # print('last_sep:',last_sep,sep_position[i][last_sep]) 

            # print('turn_ids:',turn_ids)
            # print('speaker_ids:',speaker_ids)

            local_mask[i, 0, turn_ids[i] == turn_ids[i].T] = 1.0
            local_mask[i, 0, :, (sep_positions[i][last_sep] + 1):] = 0
            # print('local_mask_index:',turn_ids[i] == turn_ids[i].T)

            sa_self_mask[i, 0, speaker_ids[i] == speaker_ids[i].T] = 1.0
            sa_self_mask[i, 0, :, (sep_positions[i][last_sep] + 1):] = 0
            # print('sa_self_mask_index', speaker_ids[i]==speaker_ids[i].T)
            
            global_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)] = 1.0 - local_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)]
            sa_cross_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)] = 1.0 - sa_self_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)]

        attention_mask = (1.0 - attention_mask) * -10000.0
        local_mask = (1.0 - local_mask) * -10000.0
        global_mask = (1.0 - global_mask) * -10000.0
        sa_self_mask = (1.0 - sa_self_mask) * -10000.0
        sa_cross_mask = (1.0 - sa_cross_mask) * -10000.0
        
        for i in range(speaker_ids_bert.size(0)):
            for j in range(speaker_ids_bert.size(1)):
                if speaker_ids_bert[i][j]==-2 or speaker_ids_bert[i][j]==-1:
                    speaker_ids_bert[i][j]=0
                else:
                    speaker_ids_bert[i][j]+=1
        # print("speaker_ids_bert:",speaker_ids_bert)


        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=original_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            speaker_ids=speaker_ids_bert
        )

        sequence_output = discriminator_hidden_states[0]

        local_word_level = self.localMHA[0](sequence_output, sequence_output, attention_mask = local_mask)[0]
        global_word_level = self.globalMHA[0](sequence_output, sequence_output, attention_mask = global_mask)[0]
        sa_self_word_level = self.SASelfMHA[0](sequence_output, sequence_output, attention_mask = sa_self_mask)[0]
        sa_cross_word_level = self.SACrossMHA[0](sequence_output, sequence_output, attention_mask = sa_cross_mask)[0]

        for t in range(1, self.num_decoupling):
            local_word_level = self.localMHA[t](local_word_level, local_word_level, attention_mask = local_mask)[0]
            global_word_level = self.globalMHA[t](global_word_level, global_word_level, attention_mask = global_mask)[0]
            sa_self_word_level = self.SASelfMHA[t](sa_self_word_level, sa_self_word_level, attention_mask = sa_self_mask)[0]
            sa_cross_word_level = self.SACrossMHA[t](sa_cross_word_level, sa_cross_word_level, attention_mask = sa_cross_mask)[0]

        context_word_level = self.fuse1(sequence_output, local_word_level, global_word_level)
        sa_word_level = self.fuse2(sequence_output, sa_self_word_level, sa_cross_word_level)

        word_level = torch.cat((context_word_level, sa_word_level), 2)
        word_level = self.dropout(word_level) #12.1 add a dropout

        logits = self.qa_outputs(word_level)
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
            output = (
                start_logits,
                end_logits,
                has_log
            ) + discriminator_hidden_states[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            has_log=has_log,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


# @add_start_docstrings(
#     """
#     Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
#     layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
#     """,
#     ROBERTA_START_DOCSTRING,
# )
class RobertaForSpeakerMask(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config, num_decoupling=1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_decoupling = num_decoupling
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        self.localMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.globalMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.SASelfMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.SACrossMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])

        self.fuse1 = FuseLayer(config)
        self.fuse2 = FuseLayer(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.has_ans = nn.Linear(config.hidden_size, 2)

        self.qa_outputs = nn.Linear(2*config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="roberta-base",
    #     output_type=QuestionAnsweringModelOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
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
        cls_index=None,
        speaker_ids=None,
        turn_ids=None,
        sep_positions=None,
        is_impossibles=None
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print('input_ids:',input_ids.size())
        # print('attention_mask:',attention_mask)
        # print('token_type_ids:',token_type_ids.size())
        speaker_ids_bert = speaker_ids
        turn_ids = turn_ids.unsqueeze(-1).repeat([1,1,turn_ids.size(1)])
        speaker_ids = speaker_ids.unsqueeze(-1).repeat([1,1,speaker_ids.size(1)])

        original_attention_mask = attention_mask
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        local_mask = torch.zeros_like(attention_mask, dtype = self.dtype)
        local_mask = local_mask.repeat((1,1,attention_mask.size(-1), 1)) #(batch_size * num_choice, 1, seq_len, seq_len)
        global_mask = torch.zeros_like(local_mask, dtype = self.dtype)
        sa_self_mask = torch.zeros_like(local_mask, dtype = self.dtype)
        sa_cross_mask = torch.zeros_like(local_mask, dtype = self.dtype)

        last_seps = []

        for i in range(input_ids.size(0)):
            last_sep = 1

            while last_sep < len(sep_positions[i]) and sep_positions[i][last_sep] != 0: 
                last_sep += 1
            
            last_sep = last_sep - 1
            last_seps.append(last_sep)
            # print('last_sep:',last_sep,sep_position[i][last_sep]) 

            # print('turn_ids:',turn_ids)
            # print('speaker_ids:',speaker_ids)

            local_mask[i, 0, turn_ids[i] == turn_ids[i].T] = 1.0
            local_mask[i, 0, :, (sep_positions[i][last_sep] + 1):] = 0
            # print('local_mask_index:',turn_ids[i] == turn_ids[i].T)

            sa_self_mask[i, 0, speaker_ids[i] == speaker_ids[i].T] = 1.0
            sa_self_mask[i, 0, :, (sep_positions[i][last_sep] + 1):] = 0
            # print('sa_self_mask_index', speaker_ids[i]==speaker_ids[i].T)
            
            global_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)] = 1.0 - local_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)]
            sa_cross_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)] = 1.0 - sa_self_mask[i, 0, :, :(sep_positions[i][last_sep] + 1)]

        attention_mask = (1.0 - attention_mask) * -10000.0
        local_mask = (1.0 - local_mask) * -10000.0
        global_mask = (1.0 - global_mask) * -10000.0
        sa_self_mask = (1.0 - sa_self_mask) * -10000.0
        sa_cross_mask = (1.0 - sa_cross_mask) * -10000.0
        
        for i in range(speaker_ids_bert.size(0)):
            for j in range(speaker_ids_bert.size(1)):
                if speaker_ids_bert[i][j]==-2 or speaker_ids_bert[i][j]==-1:
                    speaker_ids_bert[i][j]=0
                else:
                    speaker_ids_bert[i][j]+=1
        # print("speaker_ids_bert:",speaker_ids_bert)

        outputs = self.roberta(
            input_ids,
            attention_mask=original_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            speaker_ids=speaker_ids_bert
        )

        sequence_output = outputs[0]

        local_word_level = self.localMHA[0](sequence_output, sequence_output, attention_mask = local_mask)[0]
        global_word_level = self.globalMHA[0](sequence_output, sequence_output, attention_mask = global_mask)[0]
        sa_self_word_level = self.SASelfMHA[0](sequence_output, sequence_output, attention_mask = sa_self_mask)[0]
        sa_cross_word_level = self.SACrossMHA[0](sequence_output, sequence_output, attention_mask = sa_cross_mask)[0]

        for t in range(1, self.num_decoupling):
            local_word_level = self.localMHA[t](local_word_level, local_word_level, attention_mask = local_mask)[0]
            global_word_level = self.globalMHA[t](global_word_level, global_word_level, attention_mask = global_mask)[0]
            sa_self_word_level = self.SASelfMHA[t](sa_self_word_level, sa_self_word_level, attention_mask = sa_self_mask)[0]
            sa_cross_word_level = self.SACrossMHA[t](sa_cross_word_level, sa_cross_word_level, attention_mask = sa_cross_mask)[0]

        # print('local_word_level:', local_word_level.size())
        # print('global_word_level:',global_word_level.size())
        # print('sa_self_word_level:',sa_self_word_level.size())
        # print('sa_cross_word_level:',sa_cross_word_level.size())

        context_word_level = self.fuse1(sequence_output, local_word_level, global_word_level)
        sa_word_level = self.fuse2(sequence_output, sa_self_word_level, sa_cross_word_level)
        # print('context_word_level:',context_word_level.size())
        # print('sa_word_level:',sa_word_level.size())

        word_level = torch.cat((context_word_level, sa_word_level), 2)
        word_level = self.dropout(word_level) #12.1 add a dropout
        # print('word_level:',word_level.size())


        logits = self.qa_outputs(word_level)
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

