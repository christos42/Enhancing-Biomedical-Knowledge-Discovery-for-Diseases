import sys
import torch
from transformers import AutoTokenizer, AutoModel


class LMREA(torch.nn.Module):
    def __init__(self, args, device):
        super(LMREA, self).__init__()

        self.args = args
        self.device = device
        if args.embed_mode == 'PubMedBERT_base':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            # Freeze the encoding layers
            modules = [self.model.embeddings, *self.model.encoder.layer[:12]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        elif args.embed_mode == 'PubMedBERT_large':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            # Freeze the encoding layers
            modules = [self.model.embeddings, *self.model.encoder.layer[:24]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        self.dropout = torch.nn.Dropout(args.dropout)
        if args.embed_mode == 'PubMedBERT_base':
            classification_input_size = 768
        elif args.embed_mode == 'PubMedBERT_large':
            classification_input_size = 1024

        if args.exp_setting == 'binary':
            classification_output_size = 1
        elif args.exp_setting == 'multi_class':
            classification_output_size = 4

        self.BN = torch.nn.BatchNorm1d(classification_input_size)
        self.classification_layer = torch.nn.Linear(classification_input_size, classification_output_size)


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

        hidden_states = x[2][1:]

        rel_representations = []
        for i, r1 in enumerate(hidden_states[self.args.encoding_layer]):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            if self.args.aggregation == 'ent_context_ent_context':
                # embeddings: ent_context_ent_context
                ent_1_rep = torch.mean(r1[start_ent_1:end_ent_1 + 1], 0)
                ent_2_rep = torch.mean(r1[start_ent_2:end_ent_2 + 1], 0)
                final_rep = torch.add(ent_1_rep, ent_2_rep)
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'atlop_context_vector':
                # extract attentions from the model output
                attentions = x['attentions'][self.args.encoding_layer][i]

                # extract hidden_states from the model output
                #hidden_states = output.last_hidden_state

                # extract attentions of the two entities and sequence
                head_attentions = torch.mean(attentions[:, start_ent_1:end_ent_1 + 1, :], 1)
                tail_attentions = torch.mean(attentions[:, start_ent_2:end_ent_2 + 1, :], 1)

                # hadamard product of the head_attentions and tail_attentions, then average over heads
                head_tail_attentions = (head_attentions * tail_attentions).mean(dim=0)

                # normalize in order to have a distribution over sequence
                head_tail_attentions /= (head_tail_attentions.sum(dim=0, keepdim=True) + torch.finfo(head_tail_attentions.dtype).eps)

                # use the head_tail_attentions distribution to aggregate info from hidden_states
                head_tail_context_vector = head_tail_attentions @ r1
                #print(head_tail_context_vector.shape)

                # Averaged representations of the entities
                ent_1_rep = torch.mean(r1[start_ent_1:end_ent_1 + 1], 0)
                ent_2_rep = torch.mean(r1[start_ent_2:end_ent_2 + 1], 0)

                final_rep = torch.add(ent_1_rep, ent_2_rep)
                final_rep = torch.add(final_rep, head_tail_context_vector)

                rel_representations.append(final_rep)
            elif self.args.aggregation == 'atlop_context_vector_only':
                # extract attentions from the model output
                attentions = x['attentions'][self.args.encoding_layer][i]

                # extract hidden_states from the model output
                #hidden_states = output.last_hidden_state

                # extract attentions of the two entities and sequence
                head_attentions = torch.mean(attentions[:, start_ent_1:end_ent_1 + 1, :], 1)
                tail_attentions = torch.mean(attentions[:, start_ent_2:end_ent_2 + 1, :], 1)

                # hadamard product of the head_attentions and tail_attentions, then average over heads
                head_tail_attentions = (head_attentions * tail_attentions).mean(dim=0)

                # normalize in order to have a distribution over sequence
                head_tail_attentions /= (head_tail_attentions.sum(dim=0, keepdim=True) + torch.finfo(head_tail_attentions.dtype).eps)

                # use the head_tail_attentions distribution to aggregate info from hidden_states
                head_tail_context_vector = head_tail_attentions @ r1
                #print(head_tail_context_vector.shape)

                rel_representations.append(head_tail_context_vector)

        rel_representations_tensor = torch.stack(rel_representations, 0)

        if self.args.do_train:
            rel_representations_tensor = self.dropout(rel_representations_tensor)

        y = self.BN(rel_representations_tensor)
        y = self.classification_layer(y)

        return y


class LMREA_proj(torch.nn.Module):
    def __init__(self, args, device):
        super(LMREA_proj, self).__init__()

        self.args = args
        self.device = device
        if args.embed_mode == 'PubMedBERT_base':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            # Freeze the encoding layers
            modules = [self.model.embeddings, *self.model.encoder.layer[:12]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        elif args.embed_mode == 'PubMedBERT_large':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            # Freeze the encoding layers
            modules = [self.model.embeddings, *self.model.encoder.layer[:24]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        self.dropout = torch.nn.Dropout(args.dropout)
        if args.embed_mode == 'PubMedBERT_base':
            classification_input_size = 768
        elif args.embed_mode == 'PubMedBERT_large':
            classification_input_size = 1024

        if args.exp_setting == 'binary':
            classification_output_size = 1
        elif args.exp_setting == 'multi_class':
            classification_output_size = 4

        self.BN = torch.nn.BatchNorm1d(classification_input_size)
        self.head_projector = torch.nn.Linear(classification_input_size, classification_input_size)
        self.tail_projector = torch.nn.Linear(classification_input_size, classification_input_size)
        self.head_tail_projector = torch.nn.Linear(classification_input_size, classification_input_size)
        self.classification_layer = torch.nn.Linear(classification_input_size, classification_output_size)


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

        hidden_states = x[2][1:]

        rel_representations = []
        for i, r1 in enumerate(hidden_states[self.args.encoding_layer]):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            if self.args.aggregation == 'ent_context_ent_context':
                # embeddings: ent_context_ent_context
                ent_1_rep = torch.mean(r1[start_ent_1:end_ent_1 + 1], 0)
                ent_2_rep = torch.mean(r1[start_ent_2:end_ent_2 + 1], 0)
                final_rep = self.head_projector(ent_1_rep) + self.tail_projector(ent_2_rep) + self.head_projector(ent_2_rep) + self.tail_projector(ent_1_rep)
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'atlop_context_vector':
                # extract attentions from the model output
                attentions = x['attentions'][self.args.encoding_layer][i]

                # extract hidden_states from the model output
                #hidden_states = output.last_hidden_state

                # extract attentions of the two entities and sequence
                head_attentions = torch.mean(attentions[:, start_ent_1:end_ent_1 + 1, :], 1)
                tail_attentions = torch.mean(attentions[:, start_ent_2:end_ent_2 + 1, :], 1)

                # hadamard product of the head_attentions and tail_attentions, then average over heads
                head_tail_attentions = (head_attentions * tail_attentions).mean(dim=0)

                # normalize in order to have a distribution over sequence
                head_tail_attentions /= (head_tail_attentions.sum(dim=0, keepdim=True) + torch.finfo(head_tail_attentions.dtype).eps)

                # use the head_tail_attentions distribution to aggregate info from hidden_states
                head_tail_context_vector = head_tail_attentions @ r1
                #print(head_tail_context_vector.shape)

                # Averaged representations of the entities
                ent_1_rep = torch.mean(r1[start_ent_1:end_ent_1 + 1], 0)
                ent_2_rep = torch.mean(r1[start_ent_2:end_ent_2 + 1], 0)

                final_rep = (self.head_projector(ent_1_rep) + self.tail_projector(ent_1_rep) +
                             self.head_projector(ent_2_rep) + self.tail_projector(ent_2_rep) +
                             self.head_tail_projector(head_tail_context_vector))
                rel_representations.append(final_rep)
            elif self.args.aggregation == 'atlop_context_vector_only':
                # extract attentions from the model output
                attentions = x['attentions'][self.args.encoding_layer][i]

                # extract hidden_states from the model output
                #hidden_states = output.last_hidden_state

                # extract attentions of the two entities and sequence
                head_attentions = torch.mean(attentions[:, start_ent_1:end_ent_1 + 1, :], 1)
                tail_attentions = torch.mean(attentions[:, start_ent_2:end_ent_2 + 1, :], 1)

                # hadamard product of the head_attentions and tail_attentions, then average over heads
                head_tail_attentions = (head_attentions * tail_attentions).mean(dim=0)

                # normalize in order to have a distribution over sequence
                head_tail_attentions /= (head_tail_attentions.sum(dim=0, keepdim=True) + torch.finfo(head_tail_attentions.dtype).eps)

                # use the head_tail_attentions distribution to aggregate info from hidden_states
                head_tail_context_vector = head_tail_attentions @ r1
                #print(head_tail_context_vector.shape)

                final_rep = self.head_tail_projector(head_tail_context_vector)
                rel_representations.append(final_rep)

        rel_representations_tensor = torch.stack(rel_representations, 0)

        if self.args.do_train:
            rel_representations_tensor = self.dropout(rel_representations_tensor)

        y = self.BN(rel_representations_tensor)
        y = self.classification_layer(y)

        return y


class LMREM(torch.nn.Module):
    def __init__(self, args, device):
        super(LMREM, self).__init__()

        self.args = args
        self.device = device
        if args.embed_mode == 'PubMedBERT_base':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            # Freeze the encoding layers
            modules = [self.model.embeddings, *self.model.encoder.layer[:12]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        elif args.embed_mode == 'PubMedBERT_large':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            # Freeze the encoding layers
            modules = [self.model.embeddings, *self.model.encoder.layer[:24]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        self.dropout = torch.nn.Dropout(args.dropout)
        if args.embed_mode == 'PubMedBERT_base':
            classification_input_size = 768
        elif args.embed_mode == 'PubMedBERT_large':
            classification_input_size = 1024

        if args.exp_setting == 'binary':
            classification_output_size = 1
        elif args.exp_setting == 'multi_class':
            classification_output_size = 4

        self.BN = torch.nn.BatchNorm1d(classification_input_size)
        self.classification_layer = torch.nn.Linear(classification_input_size, classification_output_size)


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

        hidden_states = x[2][1:]

        rel_representations = []
        for i, r1 in enumerate(hidden_states[self.args.encoding_layer]):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            if self.args.aggregation == 'ent_context_ent_context':
                # Embeddings: 'ent_context_ent_context'
                m_ent_1 = torch.mean(r1[start_ent_1:end_ent_1 + 1], 0)
                m_ent_2 = torch.mean(r1[start_ent_2:end_ent_2 + 1], 0)
                m_ent = torch.mul(m_ent_1, m_ent_2)
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'atlop_context_vector':
                # extract attentions from the model output
                attentions = x['attentions'][self.args.encoding_layer][i]

                # extract hidden_states from the model output
                #hidden_states = output.last_hidden_state

                # extract attentions of the two entities and sequence
                head_attentions = torch.mean(attentions[:, start_ent_1:end_ent_1 + 1, :], 1)
                tail_attentions = torch.mean(attentions[:, start_ent_2:end_ent_2 + 1, :], 1)

                # hadamard product of the head_attentions and tail_attentions, then average over heads
                head_tail_attentions = (head_attentions * tail_attentions).mean(dim=0)

                # normalize in order to have a distribution over sequence
                head_tail_attentions /= (head_tail_attentions.sum(dim=0, keepdim=True) + torch.finfo(head_tail_attentions.dtype).eps)

                # use the head_tail_attentions distribution to aggregate info from hidden_states
                head_tail_context_vector = head_tail_attentions @ r1
                #print(head_tail_context_vector.shape)

                # Multiplied representations of the entities
                m_ent_1 = torch.mean(r1[start_ent_1:end_ent_1 + 1], 0)
                m_ent_2 = torch.mean(r1[start_ent_2:end_ent_2 + 1], 0)
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, head_tail_context_vector)

                rel_representations.append(m_ent)

        rel_representations_tensor = torch.stack(rel_representations, 0)

        if self.args.do_train:
            rel_representations_tensor = self.dropout(rel_representations_tensor)

        y = self.BN(rel_representations_tensor)
        y = self.classification_layer(y)

        return y


class LMREM_proj(torch.nn.Module):
    def __init__(self, args, device):
        super(LMREM_proj, self).__init__()

        self.args = args
        self.device = device
        if args.embed_mode == 'PubMedBERT_base':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            # Freeze the encoding layers
            modules = [self.model.embeddings, *self.model.encoder.layer[:12]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        elif args.embed_mode == 'PubMedBERT_large':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            # Freeze the encoding layers
            modules = [self.model.embeddings, *self.model.encoder.layer[:24]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        self.dropout = torch.nn.Dropout(args.dropout)
        if args.embed_mode == 'PubMedBERT_base':
            classification_input_size = 768
        elif args.embed_mode == 'PubMedBERT_large':
            classification_input_size = 1024

        if args.exp_setting == 'binary':
            classification_output_size = 1
        elif args.exp_setting == 'multi_class':
            classification_output_size = 4

        self.BN = torch.nn.BatchNorm1d(classification_input_size)
        self.head_projector = torch.nn.Linear(classification_input_size, classification_input_size)
        self.tail_projector = torch.nn.Linear(classification_input_size, classification_input_size)
        self.head_tail_projector = torch.nn.Linear(classification_input_size, classification_input_size)
        self.classification_layer = torch.nn.Linear(classification_input_size, classification_output_size)


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

        hidden_states = x[2][1:]

        rel_representations = []
        for i, r1 in enumerate(hidden_states[self.args.encoding_layer]):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            if self.args.aggregation == 'ent_context_ent_context':
                # Embeddings: 'ent_context_ent_context'
                m_ent_1 = torch.mean(r1[start_ent_1:end_ent_1 + 1], 0)
                m_ent_2 = torch.mean(r1[start_ent_2:end_ent_2 + 1], 0)
                m_ent_1 = self.head_projector(m_ent_1) + self.tail_projector(m_ent_1)
                m_ent_2 = self.head_projector(m_ent_2) + self.tail_projector(m_ent_2)
                m_ent = torch.mul(m_ent_1, m_ent_2)
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'atlop_context_vector':
                # extract attentions from the model output
                attentions = x['attentions'][self.args.encoding_layer][i]

                # extract hidden_states from the model output
                #hidden_states = output.last_hidden_state

                # extract attentions of the two entities and sequence
                head_attentions = torch.mean(attentions[:, start_ent_1:end_ent_1 + 1, :], 1)
                tail_attentions = torch.mean(attentions[:, start_ent_2:end_ent_2 + 1, :], 1)

                # hadamard product of the head_attentions and tail_attentions, then average over heads
                head_tail_attentions = (head_attentions * tail_attentions).mean(dim=0)

                # normalize in order to have a distribution over sequence
                head_tail_attentions /= (head_tail_attentions.sum(dim=0, keepdim=True) + torch.finfo(head_tail_attentions.dtype).eps)

                # use the head_tail_attentions distribution to aggregate info from hidden_states
                head_tail_context_vector = head_tail_attentions @ r1
                #print(head_tail_context_vector.shape)

                # Multiplied representations of the entities
                m_ent_1 = torch.mean(r1[start_ent_1:end_ent_1 + 1], 0)
                m_ent_2 = torch.mean(r1[start_ent_2:end_ent_2 + 1], 0)
                m_ent_1 = self.head_projector(m_ent_1) + self.tail_projector(m_ent_1)
                m_ent_2 = self.head_projector(m_ent_2) + self.tail_projector(m_ent_2)
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(head_tail_context_vector))

                rel_representations.append(m_ent)

        rel_representations_tensor = torch.stack(rel_representations, 0)

        if self.args.do_train:
            rel_representations_tensor = self.dropout(rel_representations_tensor)

        y = self.BN(rel_representations_tensor)
        y = self.classification_layer(rel_representations_tensor)

        return y


class LMRE_attention(torch.nn.Module):
    def __init__(self, args, device):
        super(LMRE_attention, self).__init__()

        self.args = args
        self.device = device
        if args.embed_mode == 'PubMedBERT_base':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            # Freeze the encoding layers
            modules = [self.model.embeddings, *self.model.encoder.layer[:12]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        elif args.embed_mode == 'PubMedBERT_large':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            # Freeze the encoding layers
            modules = [self.model.embeddings, *self.model.encoder.layer[:24]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        self.dropout = torch.nn.Dropout(args.dropout)
        if args.embed_mode == 'PubMedBERT_base':
            if self.args.aggregation == 'layer_specific' or self.args.aggregation == 'head_specific':
                classification_input_size = 12 + 12
            elif self.args.aggregation == 'non_specific':
                classification_input_size = 12 * 12 + 12 * 12
        elif args.embed_mode == 'PubMedBERT_large':
            if self.args.aggregation == 'layer_specific':
                classification_input_size = 16 + 16
            elif self.args.aggregation == 'head_specific':
                classification_input_size = 24 + 24
            elif self.args.aggregation == 'non_specific':
                classification_input_size = 24 * 16 + 24 * 16

        if args.exp_setting == 'binary':
            classification_output_size = 1
        elif args.exp_setting == 'multi_class':
            classification_output_size = 4

        self.BN = torch.nn.BatchNorm1d(classification_input_size)
        self.classification_layer = torch.nn.Linear(classification_input_size, classification_output_size)


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

        hidden_states = x[2][1:]

        rel_representations = []
        for i, _ in enumerate(hidden_states[-1]):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            if self.args.aggregation == 'layer_specific':
                # extract attentions from the model output
                attentions = x['attentions'][self.args.encoding_layer][i]

                # extract attentions of the two entities and sequence
                ent_1_attentions = torch.mean(torch.mean(attentions[:, start_ent_1:end_ent_1 + 1, start_ent_2:end_ent_2 + 1], 1), 1)
                ent_2_attentions = torch.mean(torch.mean(attentions[:, start_ent_2:end_ent_2 + 1, start_ent_1:end_ent_1 + 1], 1), 1)

                attentions_scores = torch.cat((ent_1_attentions, ent_2_attentions))

                rel_representations.append(attentions_scores)
            elif self.args.aggregation == 'head_specific':
                # extract attentions of a specific head from the model output
                # x['attentions']: Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length)
                attentions = []
                for layer_attentions in x['attentions']:
                    attentions.append(torch.squeeze(layer_attentions[i][self.args.attention_head], 0))

                attentions_tensor = torch.stack(attentions, 0)

                # extract attentions of the two entities and sequence
                ent_1_attentions = torch.mean(torch.mean(attentions_tensor[:, start_ent_1:end_ent_1 + 1, start_ent_2:end_ent_2 + 1], 1), 1)
                ent_2_attentions = torch.mean(torch.mean(attentions_tensor[:, start_ent_2:end_ent_2 + 1, start_ent_1:end_ent_1 + 1], 1), 1)

                attentions_scores = torch.cat((ent_1_attentions, ent_2_attentions))

                rel_representations.append(attentions_scores)
            elif self.args.aggregation == 'non_specific':
                # extract attentions of every layer and attention head from the model output
                attentions = []
                for layer_attentions in x['attentions']:
                    attentions.append(layer_attentions[i])

                attentions_tensor = torch.stack(attentions, 0)
                sec_len = attentions_tensor.shape[-1]
                attentions_tensor = attentions_tensor.view(-1, sec_len , sec_len)

                # extract attentions of the two entities and sequence
                ent_1_attentions = torch.mean(torch.mean(attentions_tensor[:, start_ent_1:end_ent_1 + 1, start_ent_2:end_ent_2 + 1], 1), 1)
                ent_2_attentions = torch.mean(torch.mean(attentions_tensor[:, start_ent_2:end_ent_2 + 1, start_ent_1:end_ent_1 + 1], 1), 1)

                attentions_scores = torch.cat((ent_1_attentions, ent_2_attentions))

                rel_representations.append(attentions_scores)

        rel_representations_tensor = torch.stack(rel_representations, 0)

        if self.args.do_train:
            rel_representations_tensor = self.dropout(rel_representations_tensor)

        y = self.BN(rel_representations_tensor)
        y = self.classification_layer(y)

        return y