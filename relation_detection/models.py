import sys
import torch
from transformers import AutoTokenizer, AutoModel, BioGptTokenizer, BioGptModel


class LaMReDA(torch.nn.Module):
    def __init__(self, args, device):
        super(LaMReDA, self).__init__()

        self.args = args
        self.device = device
        if args.embed_mode == 'BiomedBERT_base':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
            # Initialize randomly (using seed) the embeddings of the new tokens
            weights = self.model.embeddings.word_embeddings.weight.data
            torch.manual_seed(42)
            #new_weights = torch.cat((weights, torch.unsqueeze(torch.rand(768), 0)), 0)
            #new_weights = torch.cat((new_weights, torch.unsqueeze(torch.rand(768), 0)), 0)
            # Idea: small initialization embedding
            w1 = torch.empty(768)
            w1 = torch.nn.init.uniform_(w1, a=-1e-4, b=1e-4)
            w1 = torch.unsqueeze(w1, 0)
            w2 = torch.empty(768)
            w2 = torch.nn.init.uniform_(w2, a=-1e-4, b=1e-4)
            w2 = torch.unsqueeze(w2, 0)
            new_weights = torch.cat((weights, w1, w2), 0)
            new_emb = torch.nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
            self.model.embeddings.word_embeddings = new_emb

            self.start_ent_token_index = self.tokenizer.encode("[ent]", add_special_tokens=False)[0]
        elif args.embed_mode == 'BiomedBERT_large':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            # Initialize randomly (using seed) the embeddings of the new tokens
            weights = self.model.embeddings.word_embeddings.weight.data
            torch.manual_seed(42)
            #new_weights = torch.cat((weights, torch.unsqueeze(torch.rand(1024), 0)), 0)
            #new_weights = torch.cat((new_weights, torch.unsqueeze(torch.rand(1024), 0)), 0)
            # Idea: small initialization embedding
            w1 = torch.empty(1024)
            w1 = torch.nn.init.uniform_(w1, a=-1e-4, b=1e-4)
            w1 = torch.unsqueeze(w1, 0)
            w2 = torch.empty(1024)
            w2 = torch.nn.init.uniform_(w2, a=-1e-4, b=1e-4)
            w2 = torch.unsqueeze(w2, 0)
            new_weights = torch.cat((weights, w1, w2), 0)
            new_emb = torch.nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
            self.model.embeddings.word_embeddings = new_emb

            self.start_ent_token_index = self.tokenizer.encode("[ent]", add_special_tokens=False)[0]
        elif args.embed_mode == 'BioLinkBERT_base':
            #self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            self.tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            self.model = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-base")
            # Initialize randomly (using seed) the embeddings of the new tokens
            weights = self.model.embeddings.word_embeddings.weight.data
            torch.manual_seed(42)
            #new_weights = torch.cat((weights, torch.unsqueeze(torch.rand(768), 0)), 0)
            #new_weights = torch.cat((new_weights, torch.unsqueeze(torch.rand(768), 0)), 0)
            # Idea: small initialization embedding
            w1 = torch.empty(768)
            w1 = torch.nn.init.uniform_(w1, a=-1e-4, b=1e-4)
            w1 = torch.unsqueeze(w1, 0)
            w2 = torch.empty(768)
            w2 = torch.nn.init.uniform_(w2, a=-1e-4, b=1e-4)
            w2 = torch.unsqueeze(w2, 0)
            new_weights = torch.cat((weights, w1, w2), 0)
            new_emb = torch.nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
            self.model.embeddings.word_embeddings = new_emb

            self.start_ent_token_index = self.tokenizer.encode("[ent]", add_special_tokens=False)[0]
        elif args.embed_mode == 'BioLinkBERT_large':
            self.tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            self.model = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-large")
            # Initialize randomly (using seed) the embeddings of the new tokens
            weights = self.model.embeddings.word_embeddings.weight.data
            torch.manual_seed(42)
            #new_weights = torch.cat((weights, torch.unsqueeze(torch.rand(1024), 0)), 0)
            #new_weights = torch.cat((new_weights, torch.unsqueeze(torch.rand(1024), 0)), 0)
            # Idea: small initialization embedding
            w1 = torch.empty(1024)
            w1 = torch.nn.init.uniform_(w1, a=-1e-4, b=1e-4)
            w1 = torch.unsqueeze(w1, 0)
            w2 = torch.empty(1024)
            w2 = torch.nn.init.uniform_(w2, a=-1e-4, b=1e-4)
            w2 = torch.unsqueeze(w2, 0)
            new_weights = torch.cat((weights, w1, w2), 0)
            new_emb = torch.nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
            self.model.embeddings.word_embeddings = new_emb

            self.start_ent_token_index = self.tokenizer.encode("[ent]", add_special_tokens=False)[0]
        elif args.embed_mode == 'BioGPT_base':
            #self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            #self.model = BioGptModel.from_pretrained("microsoft/biogpt")
            self.model = AutoModel.from_pretrained("microsoft/biogpt")

            # Initialize randomly (using seed) the embeddings of the new tokens
            weights = self.model.embed_tokens.weight.data

            torch.manual_seed(42)
            # new_weights = torch.cat((weights, torch.unsqueeze(torch.rand(768), 0)), 0)
            # new_weights = torch.cat((new_weights, torch.unsqueeze(torch.rand(768), 0)), 0)
            # Idea: small initialization embedding
            w1 = torch.empty(1024)
            w1 = torch.nn.init.uniform_(w1, a=-1e-4, b=1e-4)
            w1 = torch.unsqueeze(w1, 0)
            w2 = torch.empty(1024)
            w2 = torch.nn.init.uniform_(w2, a=-1e-4, b=1e-4)
            w2 = torch.unsqueeze(w2, 0)
            new_weights = torch.cat((weights, w1, w2), 0)
            new_emb = torch.nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
            self.model.embed_tokens = new_emb

            self.start_ent_token_index = self.tokenizer.encode("[ent]", add_special_tokens=False)[0]
        elif args.embed_mode == 'BioGPT_large':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            self.model = AutoModel.from_pretrained("microsoft/BioGPT-Large")
            # Initialize randomly (using seed) the embeddings of the new tokens
            weights = self.model.embed_tokens.weight.data
            torch.manual_seed(42)
            # new_weights = torch.cat((weights, torch.unsqueeze(torch.rand(1024), 0)), 0)
            # new_weights = torch.cat((new_weights, torch.unsqueeze(torch.rand(1024), 0)), 0)
            # Idea: small initialization embedding
            w1 = torch.empty(1024)
            w1 = torch.nn.init.uniform_(w1, a=-1e-4, b=1e-4)
            w1 = torch.unsqueeze(w1, 0)
            w2 = torch.empty(1024)
            w2 = torch.nn.init.uniform_(w2, a=-1e-4, b=1e-4)
            w2 = torch.unsqueeze(w2, 0)
            new_weights = torch.cat((weights, w1, w2), 0)
            new_emb = torch.nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
            self.model.embed_tokens = new_emb

        self.dropout = torch.nn.Dropout(args.dropout)
        if args.embed_mode in ['BiomedBERT_base', 'BioLinkBERT_base']:
            classification_input_size = 768
        elif args.embed_mode in ['BiomedBERT_large', 'BioLinkBERT_large', 'BioGPT_base', 'BioGPT_large']:
            classification_input_size = 1024

        if args.exp_setting == 'binary':
            classification_output_size = 1
        elif args.exp_setting == 'multi_class':
            classification_output_size = 4

        if args.projection_dimension == 0:
            self.BN = torch.nn.BatchNorm1d(classification_input_size)
            self.head_projector = torch.nn.Linear(classification_input_size, classification_input_size)
            self.tail_projector = torch.nn.Linear(classification_input_size, classification_input_size)
            self.head_tail_projector = torch.nn.Linear(classification_input_size, classification_input_size)
            self.classification_layer = torch.nn.Linear(classification_input_size, classification_output_size)
        else:
            self.BN = torch.nn.BatchNorm1d(args.projection_dimension)
            self.head_projector = torch.nn.Linear(classification_input_size, args.projection_dimension)
            self.tail_projector = torch.nn.Linear(classification_input_size, args.projection_dimension)
            self.head_tail_projector = torch.nn.Linear(classification_input_size, args.projection_dimension)
            self.classification_layer = torch.nn.Linear(args.projection_dimension, classification_output_size)


    def forward(self, x, entities_range):
        x = self.tokenizer(x, return_tensors="pt",
                           padding='longest',
                           add_special_tokens=True,
                           is_split_into_words=True).to(self.device)
        input_ids = x['input_ids'].to(self.device)
        #x = self.model(**x)[0]
        x = self.model(input_ids = input_ids,
                       output_attentions = True,
                       output_hidden_states = True)

        rel_representations = []
        #for i, r1 in enumerate(x):
        for i, r1 in enumerate(x['last_hidden_state']):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            if self.args.aggregation == 'inter':
                if end_ent_1 + 1 == start_ent_2:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_2 + 1 == start_ent_1:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_1 < start_ent_2:
                    inter_rep = torch.mean(r1[end_ent_1 + 1:start_ent_2], 0)
                elif end_ent_2 < start_ent_1:
                    inter_rep = torch.mean(r1[end_ent_2 + 1:start_ent_1], 0)
                else:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)

                final_rep =  self.head_tail_projector(inter_rep)
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'start_start':
                final_rep = self.head_projector(r1[start_ent_1]) + self.tail_projector(r1[start_ent_2]) + self.head_projector(r1[start_ent_2]) + self.tail_projector(r1[start_ent_1])
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'end_end':
                final_rep = self.head_projector(r1[end_ent_1]) + self.tail_projector(r1[end_ent_2]) + self.head_projector(r1[end_ent_2]) + self.tail_projector(r1[end_ent_1])
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'ent_context_ent_context':
                ent_1_rep = torch.mean(r1[start_ent_1 + 1:end_ent_1], 0)
                ent_2_rep = torch.mean(r1[start_ent_2 + 1:end_ent_2], 0)
                final_rep = self.head_projector(ent_1_rep) + self.tail_projector(ent_2_rep) + self.head_projector(ent_2_rep) + self.tail_projector(ent_1_rep)
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'start_end_start_end':
                final_rep = self.head_projector(r1[start_ent_1]) + self.tail_projector(
                    r1[start_ent_2]) + self.head_projector(r1[start_ent_2]) + self.tail_projector(
                    r1[start_ent_1]) + self.head_projector(r1[end_ent_1]) + self.tail_projector(
                    r1[end_ent_2]) + self.head_projector(r1[end_ent_2]) + self.tail_projector(r1[end_ent_1])
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'cls_start_start':
                final_rep = self.head_projector(r1[start_ent_1]) + self.tail_projector(
                    r1[start_ent_2]) + self.head_projector(r1[start_ent_2]) + self.tail_projector(r1[start_ent_1]) + self.head_tail_projector(r1[0])
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'cls_end_end':
                final_rep = self.head_projector(r1[end_ent_1]) + self.tail_projector(
                    r1[end_ent_2]) + self.head_projector(r1[end_ent_2]) + self.tail_projector(r1[end_ent_1]) + self.head_tail_projector(r1[0])
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'cls_ent_context_ent_context':
                ent_1_rep = torch.mean(r1[start_ent_1 + 1:end_ent_1], 0)
                ent_2_rep = torch.mean(r1[start_ent_2 + 1:end_ent_2], 0)
                final_rep = self.head_projector(ent_1_rep) + self.tail_projector(ent_2_rep) + self.head_projector(
                    ent_2_rep) + self.tail_projector(ent_1_rep) + self.head_tail_projector(r1[0])
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'cls_inter':
                if end_ent_1 + 1 == start_ent_2:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_2 + 1 == start_ent_1:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_1 < start_ent_2:
                    inter_rep = torch.mean(r1[end_ent_1 + 1:start_ent_2], 0)
                elif end_ent_2 < start_ent_1:
                    inter_rep = torch.mean(r1[end_ent_2 + 1:start_ent_1], 0)
                else:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)

                final_rep = self.head_tail_projector(inter_rep) + self.head_tail_projector(r1[0])
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'cls_start_end_start_end':
                final_rep = self.head_projector(r1[start_ent_1]) + self.tail_projector(
                    r1[start_ent_2]) + self.head_projector(r1[start_ent_2]) + self.tail_projector(
                    r1[start_ent_1]) + self.head_projector(r1[end_ent_1]) + self.tail_projector(
                    r1[end_ent_2]) + self.head_projector(r1[end_ent_2]) + self.tail_projector(r1[end_ent_1]) + + self.head_tail_projector(r1[0])
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'start_inter_start':
                if end_ent_1 + 1 == start_ent_2:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_2 + 1 == start_ent_1:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_1 < start_ent_2:
                    inter_rep = torch.mean(r1[end_ent_1 + 1:start_ent_2], 0)
                elif end_ent_2 < start_ent_1:
                    inter_rep = torch.mean(r1[end_ent_2 + 1:start_ent_1], 0)
                else:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)

                final_rep = self.head_projector(r1[start_ent_1]) + self.tail_projector(
                    r1[start_ent_2]) + self.head_projector(r1[start_ent_2]) + self.tail_projector(r1[start_ent_1]) + self.head_tail_projector(inter_rep)
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'end_inter_end':
                if end_ent_1 + 1 == start_ent_2:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_2 + 1 == start_ent_1:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_1 < start_ent_2:
                    inter_rep = torch.mean(r1[end_ent_1 + 1:start_ent_2], 0)
                elif end_ent_2 < start_ent_1:
                    inter_rep = torch.mean(r1[end_ent_2 + 1:start_ent_1], 0)
                else:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)

                final_rep = self.head_projector(r1[end_ent_1]) + self.tail_projector(
                    r1[end_ent_2]) + self.head_projector(r1[end_ent_2]) + self.tail_projector(r1[end_ent_1]) + self.head_tail_projector(inter_rep)
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'start_end_inter_start_end':
                if end_ent_1 + 1 == start_ent_2:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_2 + 1 == start_ent_1:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_1 < start_ent_2:
                    inter_rep = torch.mean(r1[end_ent_1 + 1:start_ent_2], 0)
                elif end_ent_2 < start_ent_1:
                    inter_rep = torch.mean(r1[end_ent_2 + 1:start_ent_1], 0)
                else:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)

                final_rep = self.head_projector(r1[start_ent_1]) + self.tail_projector(
                    r1[start_ent_2]) + self.head_projector(r1[start_ent_2]) + self.tail_projector(
                    r1[start_ent_1]) + self.head_projector(r1[end_ent_1]) + self.tail_projector(
                    r1[end_ent_2]) + self.head_projector(r1[end_ent_2]) + self.tail_projector(r1[end_ent_1]) + self.head_tail_projector(inter_rep)
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'ent_context_inter_ent_context':
                if end_ent_1 + 1 == start_ent_2:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_2 + 1 == start_ent_1:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_1 < start_ent_2:
                    inter_rep = torch.mean(r1[end_ent_1 + 1:start_ent_2], 0)
                elif end_ent_2 < start_ent_1:
                    inter_rep = torch.mean(r1[end_ent_2 + 1:start_ent_1], 0)
                else:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                ent_1_rep = torch.mean(r1[start_ent_1 + 1:end_ent_1], 0)
                ent_2_rep = torch.mean(r1[start_ent_2 + 1:end_ent_2], 0)
                final_rep = self.head_projector(ent_1_rep) + self.tail_projector(ent_2_rep) + self.head_projector(
                    ent_2_rep) + self.tail_projector(ent_1_rep) + self.head_tail_projector(inter_rep)
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'atlop_context_vector_only':
                # check where [ent] in input ids
                #where_start_ent_in_input_ids = torch.where(input_ids[i] == self.start_ent_token_index)

                # get batch and sequence indices of the first [ent]
                #head_batch_index = where_start_ent_in_input_ids[0][0]
                #head_sequence_index = where_start_ent_in_input_ids[0][0]

                # get batch and sequence indices of the second [ent]
                #tail_batch_index = where_start_ent_in_input_ids[0][1]
                #tail_sequence_index = where_start_ent_in_input_ids[0][1]

                # extract attentions from the model output
                attentions = x['attentions'][-1][i]

                # extract hidden_states from the model output
                #hidden_states = output.last_hidden_state

                # extract attentions of first [ent] and sequence
                #head_attentions = attentions[[i], :, [head_sequence_index], :]
                #head_attentions = attentions[:, head_sequence_index, :]
                head_attentions = attentions[:, start_ent_1, :]
                # extract attentions of second [ent] and sequence
                #tail_attentions = attentions[[i], :, [tail_sequence_index], :]
                #tail_attentions = attentions[:, tail_sequence_index, :]
                tail_attentions = attentions[:, start_ent_2, :]

                # hadamard product of the head_attentions and tail_attentions, then average over heads
                #head_tail_attentions = (head_attentions * tail_attentions).mean(dim=1)
                head_tail_attentions = (head_attentions * tail_attentions).mean(dim=0)

                # normalize in order to have a distribution over sequence
                #head_tail_attentions /= (head_tail_attentions.sum(dim=1, keepdim=True) + torch.finfo(head_tail_attentions.dtype).eps)
                head_tail_attentions /= (head_tail_attentions.sum(dim=0, keepdim=True) + torch.finfo(head_tail_attentions.dtype).eps)

                # use the head_tail_attentions distribution to aggregate info from hidden_states
                #head_tail_context_vector = torch.einsum("bse,bs->be", x.last_hidden_state, head_tail_attentions)
                #head_tail_context_vector = torch.einsum("se,s->e", r1, head_tail_attentions)
                head_tail_context_vector = head_tail_attentions @ r1

                final_rep = self.head_tail_projector(head_tail_context_vector)
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'atlop_context_vector':
                # check where [ent] in input ids
                #where_start_ent_in_input_ids = torch.where(input_ids[i] == self.start_ent_token_index)

                # get batch and sequence indices of the first [ent]
                #head_batch_index = where_start_ent_in_input_ids[0][0]
                #head_sequence_index = where_start_ent_in_input_ids[0][0]

                # get batch and sequence indices of the second [ent]
                #tail_batch_index = where_start_ent_in_input_ids[0][1]
                #tail_sequence_index = where_start_ent_in_input_ids[0][1]

                # extract attentions from the model output
                #attentions = x['attentions'][-1]
                attentions = x['attentions'][-1][i]

                # extract hidden_states from the model output
                #hidden_states = output.last_hidden_state

                # extract attentions of first [ent] and sequence
                #head_attentions = attentions[[i], :, [head_sequence_index], :]
                #head_attentions = attentions[:, head_sequence_index, :]
                head_attentions = attentions[:, start_ent_1, :]
                # extract attentions of second [ent] and sequence
                #tail_attentions = attentions[[i], :, [tail_sequence_index], :]
                #tail_attentions = attentions[:, tail_sequence_index, :]
                tail_attentions = attentions[:, start_ent_2, :]

                # hadamard product of the head_attentions and tail_attentions, then average over heads
                #head_tail_attentions = (head_attentions * tail_attentions).mean(dim=1)
                head_tail_attentions = (head_attentions * tail_attentions).mean(dim=0)

                # normalize in order to have a distribution over sequence
                #head_tail_attentions /= (head_tail_attentions.sum(dim=1, keepdim=True) + torch.finfo(head_tail_attentions.dtype).eps)
                head_tail_attentions /= (head_tail_attentions.sum(dim=0, keepdim=True) + torch.finfo(head_tail_attentions.dtype).eps)

                # use the head_tail_attentions distribution to aggregate info from hidden_states
                #head_tail_context_vector = torch.einsum("bse,bs->be", x.last_hidden_state, head_tail_attentions)
                #head_tail_context_vector = torch.einsum("se,s->e", r1, head_tail_attentions)
                head_tail_context_vector = head_tail_attentions @ r1

                final_rep = self.head_projector(r1[start_ent_1]) + self.tail_projector(
                    r1[start_ent_2]) + self.head_projector(r1[start_ent_2]) + self.tail_projector(
                    r1[start_ent_1]) + self.head_tail_projector(head_tail_context_vector)
                rel_representations.append(final_rep)


        rel_representations_tensor = torch.stack(rel_representations, 0)

        if self.args.do_train:
            rel_representations_tensor = self.dropout(rel_representations_tensor)

        if rel_representations_tensor.shape[0] != 1:
            y = self.BN(rel_representations_tensor)
            y = self.classification_layer(y)
        else:
            y = self.classification_layer(rel_representations_tensor)

        return y


class LaMReDM(torch.nn.Module):
    def __init__(self, args, device):
        super(LaMReDM, self).__init__()

        self.args = args
        self.device = device
        if args.embed_mode == 'BiomedBERT_base':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
            # Initialize randomly (using seed) the embeddings of the new tokens
            weights = self.model.embeddings.word_embeddings.weight.data
            torch.manual_seed(42)
            #new_weights = torch.cat((weights, torch.unsqueeze(torch.rand(768), 0)), 0)
            #new_weights = torch.cat((new_weights, torch.unsqueeze(torch.rand(768), 0)), 0)
            # Idea: small initialization embedding
            w1 = torch.empty(768)
            w1 = torch.nn.init.uniform_(w1, a=-1e-4, b=1e-4)
            w1 = torch.unsqueeze(w1, 0)
            w2 = torch.empty(768)
            w2 = torch.nn.init.uniform_(w2, a=-1e-4, b=1e-4)
            w2 = torch.unsqueeze(w2, 0)
            new_weights = torch.cat((weights, w1, w2), 0)
            new_emb = torch.nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
            self.model.embeddings.word_embeddings = new_emb
        elif args.embed_mode == 'BiomedBERT_large':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            # Initialize randomly (using seed) the embeddings of the new tokens
            weights = self.model.embeddings.word_embeddings.weight.data
            torch.manual_seed(42)
            #new_weights = torch.cat((weights, torch.unsqueeze(torch.rand(1024), 0)), 0)
            #new_weights = torch.cat((new_weights, torch.unsqueeze(torch.rand(1024), 0)), 0)
            # Idea: small initialization embedding
            w1 = torch.empty(1024)
            w1 = torch.nn.init.uniform_(w1, a=-1e-4, b=1e-4)
            w1 = torch.unsqueeze(w1, 0)
            w2 = torch.empty(1024)
            w2 = torch.nn.init.uniform_(w2, a=-1e-4, b=1e-4)
            w2 = torch.unsqueeze(w2, 0)
            new_weights = torch.cat((weights, w1, w2), 0)
            new_emb = torch.nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
            self.model.embeddings.word_embeddings = new_emb
        elif args.embed_mode == 'BioLinkBERT_base':
            #self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            self.tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            self.model = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-base")
            # Initialize randomly (using seed) the embeddings of the new tokens
            weights = self.model.embeddings.word_embeddings.weight.data
            torch.manual_seed(42)
            #new_weights = torch.cat((weights, torch.unsqueeze(torch.rand(768), 0)), 0)
            #new_weights = torch.cat((new_weights, torch.unsqueeze(torch.rand(768), 0)), 0)
            # Idea: small initialization embedding
            w1 = torch.empty(768)
            w1 = torch.nn.init.uniform_(w1, a=-1e-4, b=1e-4)
            w1 = torch.unsqueeze(w1, 0)
            w2 = torch.empty(768)
            w2 = torch.nn.init.uniform_(w2, a=-1e-4, b=1e-4)
            w2 = torch.unsqueeze(w2, 0)
            new_weights = torch.cat((weights, w1, w2), 0)
            new_emb = torch.nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
            self.model.embeddings.word_embeddings = new_emb

            self.start_ent_token_index = self.tokenizer.encode("[ent]", add_special_tokens=False)[0]
        elif args.embed_mode == 'BioLinkBERT_large':
            self.tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            self.model = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-large")
            # Initialize randomly (using seed) the embeddings of the new tokens
            weights = self.model.embeddings.word_embeddings.weight.data
            torch.manual_seed(42)
            #new_weights = torch.cat((weights, torch.unsqueeze(torch.rand(1024), 0)), 0)
            #new_weights = torch.cat((new_weights, torch.unsqueeze(torch.rand(1024), 0)), 0)
            # Idea: small initialization embedding
            w1 = torch.empty(1024)
            w1 = torch.nn.init.uniform_(w1, a=-1e-4, b=1e-4)
            w1 = torch.unsqueeze(w1, 0)
            w2 = torch.empty(1024)
            w2 = torch.nn.init.uniform_(w2, a=-1e-4, b=1e-4)
            w2 = torch.unsqueeze(w2, 0)
            new_weights = torch.cat((weights, w1, w2), 0)
            new_emb = torch.nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
            self.model.embeddings.word_embeddings = new_emb

            self.start_ent_token_index = self.tokenizer.encode("[ent]", add_special_tokens=False)[0]
        elif args.embed_mode == 'BioGPT_base':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            self.model = AutoModel.from_pretrained("microsoft/biogpt")
            # Initialize randomly (using seed) the embeddings of the new tokens
            weights = self.model.embed_tokens.weight.data
            torch.manual_seed(42)
            # new_weights = torch.cat((weights, torch.unsqueeze(torch.rand(768), 0)), 0)
            # new_weights = torch.cat((new_weights, torch.unsqueeze(torch.rand(768), 0)), 0)
            # Idea: small initialization embedding
            w1 = torch.empty(1024)
            w1 = torch.nn.init.uniform_(w1, a=-1e-4, b=1e-4)
            w1 = torch.unsqueeze(w1, 0)
            w2 = torch.empty(1024)
            w2 = torch.nn.init.uniform_(w2, a=-1e-4, b=1e-4)
            w2 = torch.unsqueeze(w2, 0)
            new_weights = torch.cat((weights, w1, w2), 0)
            new_emb = torch.nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
            self.model.embed_tokens.weight.data = new_emb

            self.start_ent_token_index = self.tokenizer.encode("[ent]", add_special_tokens=False)[0]
        elif args.embed_mode == 'BioGPT_large':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            self.model = AutoModel.from_pretrained("microsoft/BioGPT-Large")
            # Initialize randomly (using seed) the embeddings of the new tokens
            weights = self.model.embed_tokens.weight.data
            torch.manual_seed(42)
            # new_weights = torch.cat((weights, torch.unsqueeze(torch.rand(1024), 0)), 0)
            # new_weights = torch.cat((new_weights, torch.unsqueeze(torch.rand(1024), 0)), 0)
            # Idea: small initialization embedding
            w1 = torch.empty(1024)
            w1 = torch.nn.init.uniform_(w1, a=-1e-4, b=1e-4)
            w1 = torch.unsqueeze(w1, 0)
            w2 = torch.empty(1024)
            w2 = torch.nn.init.uniform_(w2, a=-1e-4, b=1e-4)
            w2 = torch.unsqueeze(w2, 0)
            new_weights = torch.cat((weights, w1, w2), 0)
            new_emb = torch.nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
            self.model.embed_tokens = new_emb

        self.dropout = torch.nn.Dropout(args.dropout)
        if args.embed_mode in ['BiomedBERT_base', 'BioLinkBERT_base']:
            classification_input_size = 768
        elif args.embed_mode in ['BiomedBERT_large', 'BioLinkBERT_large', 'BioGPT_base', 'BioGPT_large']:
            classification_input_size = 1024

        if args.exp_setting == 'binary':
            classification_output_size = 1
        elif args.exp_setting == 'multi_class':
            classification_output_size = 4

        if args.projection_dimension == 0:
            self.BN = torch.nn.BatchNorm1d(classification_input_size)
            self.head_projector = torch.nn.Linear(classification_input_size, classification_input_size)
            self.tail_projector = torch.nn.Linear(classification_input_size, classification_input_size)
            self.head_tail_projector = torch.nn.Linear(classification_input_size, classification_input_size)
            self.classification_layer = torch.nn.Linear(classification_input_size, classification_output_size)
        else:
            self.BN = torch.nn.BatchNorm1d(args.projection_dimension)
            self.head_projector = torch.nn.Linear(classification_input_size, args.projection_dimension)
            self.tail_projector = torch.nn.Linear(classification_input_size, args.projection_dimension)
            self.head_tail_projector = torch.nn.Linear(classification_input_size, args.projection_dimension)
            self.classification_layer = torch.nn.Linear(args.projection_dimension, classification_output_size)


    def forward(self, x, entities_range):
        x = self.tokenizer(x, return_tensors="pt",
                           padding='longest',
                           add_special_tokens=True,
                           is_split_into_words=True).to(self.device)
        input_ids = x['input_ids'].to(self.device)
        # x = self.model(**x)[0]
        x = self.model(input_ids=input_ids,
                       output_attentions=True,
                       output_hidden_states=True)

        rel_representations = []
        # for i, r1 in enumerate(x):
        for i, r1 in enumerate(x['last_hidden_state']):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            if self.args.aggregation == 'start_start':
                m_ent_1 = self.head_projector(r1[start_ent_1]) + self.tail_projector(r1[start_ent_1])
                m_ent_2 = self.head_projector(r1[start_ent_2]) + self.tail_projector(r1[start_ent_2])
                m_ent = torch.mul(m_ent_1, m_ent_2)
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'end_end':
                m_ent_1 = self.head_projector(r1[end_ent_1]) + self.tail_projector(r1[end_ent_1])
                m_ent_2 = self.head_projector(r1[end_ent_2]) + self.tail_projector(r1[end_ent_2])
                m_ent = torch.mul(m_ent_1, m_ent_2)
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'start_end_start_end':
                m_ent_1 = torch.mul(self.head_projector(r1[start_ent_1]) + self.tail_projector(r1[start_ent_1]), self.head_projector(r1[end_ent_1]) + self.tail_projector(r1[end_ent_1]))
                m_ent_2 = torch.mul(self.head_projector(r1[start_ent_2]) + self.tail_projector(r1[start_ent_2]), self.head_projector(r1[end_ent_2]) + self.tail_projector(r1[end_ent_2]))
                m_ent = torch.mul(m_ent_1, m_ent_2)
                rel_representations.append(m_ent)
            if self.args.aggregation == 'cls_start_start':
                m_ent_1 = self.head_projector(r1[start_ent_1]) + self.tail_projector(r1[start_ent_1])
                m_ent_2 = self.head_projector(r1[start_ent_2]) + self.tail_projector(r1[start_ent_2])
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(r1[0]))
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'cls_end_end':
                m_ent_1 = self.head_projector(r1[end_ent_1]) + self.tail_projector(r1[end_ent_1])
                m_ent_2 = self.head_projector(r1[end_ent_2]) + self.tail_projector(r1[end_ent_2])
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(r1[0]))
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'cls_start_end_start_end':
                m_ent_1 = torch.mul(self.head_projector(r1[start_ent_1]) + self.tail_projector(r1[start_ent_1]),
                                    self.head_projector(r1[end_ent_1]) + self.tail_projector(r1[end_ent_1]))
                m_ent_2 = torch.mul(self.head_projector(r1[start_ent_2]) + self.tail_projector(r1[start_ent_2]),
                                    self.head_projector(r1[end_ent_2]) + self.tail_projector(r1[end_ent_2]))
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(r1[0]))
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'start_inter_start':
                if end_ent_1 + 1 == start_ent_2:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_2 + 1 == start_ent_1:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_1 < start_ent_2:
                    inter_rep = torch.mean(r1[end_ent_1 + 1:start_ent_2], 0)
                elif end_ent_2 < start_ent_1:
                    inter_rep = torch.mean(r1[end_ent_2 + 1:start_ent_1], 0)
                else:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                m_ent_1 = self.head_projector(r1[start_ent_1]) + self.tail_projector(
                    r1[start_ent_1])
                m_ent_2 = self.head_projector(r1[start_ent_2]) + self.tail_projector(
                    r1[start_ent_2])
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(inter_rep))
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'end_inter_end':
                if end_ent_1 + 1 == start_ent_2:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_2 + 1 == start_ent_1:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_1 < start_ent_2:
                    inter_rep = torch.mean(r1[end_ent_1 + 1:start_ent_2], 0)
                elif end_ent_2 < start_ent_1:
                    inter_rep = torch.mean(r1[end_ent_2 + 1:start_ent_1], 0)
                else:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                m_ent_1 = self.head_projector(r1[end_ent_1]) + self.tail_projector(
                    r1[end_ent_1])
                m_ent_2 = self.head_projector(r1[end_ent_2]) + self.tail_projector(
                    r1[end_ent_2])
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(inter_rep))
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'start_end_inter_start_end':
                if end_ent_1 + 1 == start_ent_2:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_2 + 1 == start_ent_1:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_1 < start_ent_2:
                    inter_rep = torch.mean(r1[end_ent_1 + 1:start_ent_2], 0)
                elif end_ent_2 < start_ent_1:
                    inter_rep = torch.mean(r1[end_ent_2 + 1:start_ent_1], 0)
                else:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                m_ent_1 = torch.mul(self.head_projector(r1[start_ent_1]) + self.tail_projector(r1[start_ent_1]),
                                    self.head_projector(r1[end_ent_1]) + self.tail_projector(r1[end_ent_1]))
                m_ent_2 = torch.mul(self.head_projector(r1[start_ent_2]) + self.tail_projector(r1[start_ent_2]),
                                    self.head_projector(r1[end_ent_2]) + self.tail_projector(r1[end_ent_2]))
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(inter_rep))
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'cls_inter':
                if end_ent_1 + 1 == start_ent_2:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_2 + 1 == start_ent_1:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_1 < start_ent_2:
                    inter_rep = torch.mean(r1[end_ent_1 + 1:start_ent_2], 0)
                elif end_ent_2 < start_ent_1:
                    inter_rep = torch.mean(r1[end_ent_2 + 1:start_ent_1], 0)
                else:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                m_ent = torch.mul(self.head_tail_projector(r1[0]), self.head_tail_projector(inter_rep))
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'ent_context_ent_context':
                m_ent_1 = torch.mean(r1[start_ent_1 + 1:end_ent_1], 0)
                m_ent_2 = torch.mean(r1[start_ent_2 + 1:end_ent_2], 0)
                m_ent_1 = self.head_projector(m_ent_1) + self.tail_projector(m_ent_1)
                m_ent_2 = self.head_projector(m_ent_2) + self.tail_projector(m_ent_2)
                m_ent = torch.mul(m_ent_1, m_ent_2)
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'cls_ent_context_ent_context':
                m_ent_1 = torch.mean(r1[start_ent_1 + 1:end_ent_1], 0)
                m_ent_2 = torch.mean(r1[start_ent_2 + 1:end_ent_2], 0)
                m_ent_1 = self.head_projector(m_ent_1) + self.tail_projector(m_ent_1)
                m_ent_2 = self.head_projector(m_ent_2) + self.tail_projector(m_ent_2)
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(r1[0]))
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'ent_context_inter_ent_context':
                if end_ent_1 + 1 == start_ent_2:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_2 + 1 == start_ent_1:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                elif end_ent_1 < start_ent_2:
                    inter_rep = torch.mean(r1[end_ent_1 + 1:start_ent_2], 0)
                elif end_ent_2 < start_ent_1:
                    inter_rep = torch.mean(r1[end_ent_2 + 1:start_ent_1], 0)
                else:
                    inter_rep = torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
                m_ent_1 = torch.mean(r1[start_ent_1 + 1:end_ent_1], 0)
                m_ent_2 = torch.mean(r1[start_ent_2 + 1:end_ent_2], 0)
                m_ent_1 = self.head_projector(m_ent_1) + self.tail_projector(m_ent_1)
                m_ent_2 = self.head_projector(m_ent_2) + self.tail_projector(m_ent_2)
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(inter_rep))
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'atlop_context_vector':
                attentions = x['attentions'][-1][i]
                head_attentions = attentions[:, start_ent_1, :]
                tail_attentions = attentions[:, start_ent_2, :]

                # hadamard product of the head_attentions and tail_attentions, then average over heads
                head_tail_attentions = (head_attentions * tail_attentions).mean(dim=0)
                #print(head_tail_attentions.shape)

                # normalize in order to have a distribution over sequence
                head_tail_attentions /= (head_tail_attentions.sum(dim=0, keepdim=True) + torch.finfo(head_tail_attentions.dtype).eps)

                # use the head_tail_attentions distribution to aggregate info from hidden_states
                head_tail_context_vector = head_tail_attentions @ r1

                m_ent_1 = torch.mean(r1[start_ent_1 + 1:end_ent_1], 0)
                m_ent_2 = torch.mean(r1[start_ent_2 + 1:end_ent_2], 0)
                m_ent_1 = self.head_projector(m_ent_1) + self.tail_projector(m_ent_1)
                m_ent_2 = self.head_projector(m_ent_2) + self.tail_projector(m_ent_2)
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(head_tail_context_vector))
                rel_representations.append(m_ent)


        rel_representations_tensor = torch.stack(rel_representations, 0)

        if self.args.do_train:
            rel_representations_tensor = self.dropout(rel_representations_tensor)

        if rel_representations_tensor.shape[0] != 1:
            y = self.BN(rel_representations_tensor)
            y = self.classification_layer(y)
        else:
            y = self.classification_layer(rel_representations_tensor)

        return y