from functools import partial
from .vision.vit import VisionTransformer
from .xbert import BertConfig, BertModel, BertLMHeadModel
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class MUMC_VQA(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 text_decoder=None,
                 tokenizer=None,
                 config=None,
                 ):
        super().__init__()

    # Initialize tokenizer properly (critical fix!)
        if tokenizer is None or isinstance(tokenizer, bool):  # Handle missing/broken tokenizer
            from transformers import AutoTokenizer  # <-- Add this import
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # <-- Default tokenizer
        else:
            self.tokenizer = tokenizer  # Use the provided tokenizer if valid
            # self.tokenizer = tokenizer

        self.distill = config['distill']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        config_encoder = BertConfig.from_json_file(config['bert_config'])
        config_decoder = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)

        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        self.text_decoder = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=config_encoder,
                                                            add_pooling_layer=False)
            self.text_decoder_m = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)
            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.text_decoder, self.text_decoder_m],
                                ]
            self.copy_params()
            self.momentum = 0.995

    def forward(self, image, question, answer=None, alpha=0, k=None, train=True):
        # 图像编码
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(image.device)
        answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device)

        # train
        if train:
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)
            if self.distill:
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image)
                    question_output_m = self.text_encoder_m(question.input_ids,
                                                            attention_mask=question.attention_mask,
                                                            encoder_hidden_states=image_embeds_m,
                                                            encoder_attention_mask=image_atts,
                                                            return_dict=True)

                    logits_m = self.text_decoder_m(answer.input_ids,
                                                   attention_mask=answer.attention_mask,
                                                   encoder_hidden_states=question_output_m.last_hidden_state,
                                                   encoder_attention_mask=question.attention_mask,
                                                   return_logits=True,
                                                   )

                answer_output = self.text_decoder(answer.input_ids,
                                                  attention_mask=answer.attention_mask,
                                                  encoder_hidden_states=question_output.last_hidden_state,
                                                  encoder_attention_mask=question.attention_mask,
                                                  labels=answer_targets,
                                                  return_dict=True,
                                                  soft_labels=F.softmax(logits_m, dim=-1),
                                                  alpha=alpha,
                                                  reduction='none',
                                                  )
            else:
                answer_output = self.text_decoder(answer.input_ids,
                                                  attention_mask=answer.attention_mask,
                                                  encoder_hidden_states=question_output.last_hidden_state,
                                                  encoder_attention_mask=question.attention_mask,
                                                  labels=answer_targets,
                                                  return_dict=True,
                                                  reduction='none',
                                                  )
            loss = answer_output.loss
            loss = loss.sum() / image.size(0)   # image.size(0) = batch_size
            return loss

        # test
        else:
            question_output = self.text_encoder(question.input_ids,
                                                attention_mask=question.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True)
            topk_ids, topk_probs = self.rank_answer(question_output.last_hidden_state, question.attention_mask,
                                                    answer.input_ids, answer.attention_mask, k)

            return topk_ids, topk_probs

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):

        num_ques = question_states.size(0)  # num_ques = batch_size_test
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(start_ids,
                                         encoder_hidden_states=question_states,
                                         encoder_attention_mask=question_atts,
                                         return_dict=True,
                                         reduction='none')
        logits = start_output.logits[:, 0, :]

        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)

        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))

        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id,
                                            -100)

        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(input_ids,
                                   attention_mask=input_atts,
                                   encoder_hidden_states=question_states,
                                   encoder_attention_mask=question_atts,
                                   labels=targets_ids,
                                   return_dict=True,
                                   reduction='none')

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)


        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)
        topk_probs = F.softmax(log_probs_sum, dim=-1)

        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)
        return topk_ids, topk_probs


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))
