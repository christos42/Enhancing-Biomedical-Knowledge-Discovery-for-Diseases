import sys
import torch
from transformers import AutoTokenizer, AutoModel


class LMCE(torch.nn.Module):
    def __init__(self, args, device):
        super(LMCE, self).__init__()

        self.args = args
        self.device = device
        if args.embed_mode == 'PubMedBERT_base':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
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
        elif args.embed_mode == 'PubMedBERT_large':
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

        self.dropout = torch.nn.Dropout(args.dropout)
        if args.embed_mode == 'PubMedBERT_base':
            if args.aggregation in ['start_start', 'end_end']:
                classification_input_size = 1536
            elif args.aggregation in ['cls_start_start', 'cls_end_end', 'start_inter_start', 'end_inter_end']:
                classification_input_size = 2304
            elif args.aggregation == 'start_end_start_end':
                classification_input_size = 3072
            elif args.aggregation in ['cls_start_end_start_end', 'start_end_inter_start_end']:
                classification_input_size = 3840
        elif args.embed_mode == 'PubMedBERT_large':
            if args.aggregation in ['start_start', 'end_end']:
                classification_input_size = 2048
            elif args.aggregation in ['cls_start_start', 'cls_end_end', 'start_inter_start', 'end_inter_end']:
                classification_input_size = 3072
            elif args.aggregation == 'start_end_start_end':
                classification_input_size = 4096
            elif args.aggregation in ['cls_start_end_start_end', 'start_end_inter_start_end']:
                classification_input_size = 5120

        if args.exp_setting == 'binary':
            classification_output_size = 1
        elif args.exp_setting == 'multi_class':
            classification_output_size = 4

        # Bi-Linear layer?
        self.classification_layer = torch.nn.Linear(classification_input_size, classification_output_size)


    def forward(self, x, entities_range):
        x = self.tokenizer(x, return_tensors="pt",
                           padding='longest',
                           add_special_tokens=True,
                           is_split_into_words=True).to(self.device)
        x = self.model(**x)[0]

        rel_representations = []
        for i, r1 in enumerate(x):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            if self.args.aggregation == 'start_start':
                if end_ent_1 < start_ent_2:
                    rel_rep = torch.cat((torch.unsqueeze(r1[start_ent_1], 0),
                                         torch.unsqueeze(r1[start_ent_2], 0)), 1)
                else:
                    rel_rep = torch.cat((torch.unsqueeze(r1[start_ent_2], 0),
                                         torch.unsqueeze(r1[start_ent_1], 0)), 1)
                #rel_representations.append(rel_rep)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'end_end':
                if end_ent_1 < start_ent_2:
                    rel_rep = torch.cat((torch.unsqueeze(r1[end_ent_1], 0),
                                         torch.unsqueeze(r1[end_ent_2], 0)), 1)
                else:
                    rel_rep = torch.cat((torch.unsqueeze(r1[end_ent_2], 0),
                                         torch.unsqueeze(r1[end_ent_1], 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'start_end_start_end':
                if end_ent_1 < start_ent_2:
                    rel_rep = torch.cat((torch.unsqueeze(r1[start_ent_1], 0),
                                         torch.unsqueeze(r1[end_ent_1], 0),
                                         torch.unsqueeze(r1[start_ent_2], 0),
                                         torch.unsqueeze(r1[end_ent_2], 0)), 1)
                else:
                    rel_rep = torch.cat((torch.unsqueeze(r1[start_ent_2], 0),
                                         torch.unsqueeze(r1[end_ent_2], 0),
                                         torch.unsqueeze(r1[start_ent_1], 0),
                                         torch.unsqueeze(r1[end_ent_1], 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            if self.args.aggregation == 'cls_start_start':
                if end_ent_1 < start_ent_2:
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(r1[start_ent_1], 0),
                                         torch.unsqueeze(r1[start_ent_2], 0)), 1)
                else:
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(r1[start_ent_2], 0),
                                         torch.unsqueeze(r1[start_ent_1], 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'cls_end_end':
                if end_ent_1 < start_ent_2:
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(r1[end_ent_1], 0),
                                         torch.unsqueeze(r1[end_ent_2], 0)), 1)
                else:
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(r1[end_ent_2], 0),
                                         torch.unsqueeze(r1[end_ent_1], 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'cls_start_end_start_end':
                if end_ent_1 < start_ent_2:
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(r1[start_ent_1], 0),
                                         torch.unsqueeze(r1[end_ent_1], 0),
                                         torch.unsqueeze(r1[start_ent_2], 0),
                                         torch.unsqueeze(r1[end_ent_2], 0)), 1)
                else:
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(r1[start_ent_2], 0),
                                         torch.unsqueeze(r1[end_ent_2], 0),
                                         torch.unsqueeze(r1[start_ent_1], 0),
                                         torch.unsqueeze(r1[end_ent_1], 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'start_inter_start':
                if end_ent_1 < start_ent_2:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_1+1:start_ent_2], 0), 0)
                    rel_rep = torch.cat((torch.unsqueeze(r1[start_ent_1], 0), inter_rep, torch.unsqueeze(r1[start_ent_2], 0)), 1)
                else:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_2+1:start_ent_1], 0), 0)
                    rel_rep = torch.cat((torch.unsqueeze(r1[start_ent_2], 0), inter_rep, torch.unsqueeze(r1[start_ent_1], 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'end_inter_end':
                if end_ent_1 < start_ent_2:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_1+1:start_ent_2], 0), 0)
                    rel_rep = torch.cat((torch.unsqueeze(r1[end_ent_1], 0),
                                         inter_rep,
                                         torch.unsqueeze(r1[end_ent_2], 0)), 1)
                else:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_2+1:start_ent_1], 0), 0)
                    rel_rep = torch.cat((torch.unsqueeze(r1[end_ent_2], 0),
                                         inter_rep,
                                         torch.unsqueeze(r1[end_ent_1], 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'start_end_inter_start_end':
                if end_ent_1 < start_ent_2:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_1+1:start_ent_2], 0), 0)
                    rel_rep = torch.cat((torch.unsqueeze(r1[start_ent_1], 0),
                                         torch.unsqueeze(r1[end_ent_1], 0),
                                         inter_rep,
                                         torch.unsqueeze(r1[start_ent_2], 0),
                                         torch.unsqueeze(r1[end_ent_2], 0)), 1)
                else:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_2+1:start_ent_1], 0), 0)
                    rel_rep = torch.cat((torch.unsqueeze(r1[start_ent_2], 0),
                                         torch.unsqueeze(r1[end_ent_2], 0),
                                         inter_rep,
                                         torch.unsqueeze(r1[start_ent_1], 0),
                                         torch.unsqueeze(r1[end_ent_1], 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))

        rel_representations_tensor = torch.stack(rel_representations, 0)

        if self.args.do_train:
            rel_representations_tensor = self.dropout(rel_representations_tensor)

        y = self.classification_layer(rel_representations_tensor)

        return y


class LMCE_mul(torch.nn.Module):
    def __init__(self, args, device):
        super(LMCE_mul, self).__init__()

        self.args = args
        self.device = device
        if args.embed_mode == 'PubMedBERT_base':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
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
        elif args.embed_mode == 'PubMedBERT_large':
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

        self.dropout = torch.nn.Dropout(args.dropout)
        if args.embed_mode == 'PubMedBERT_base':
            if args.aggregation in ['start_start', 'end_end']:
                classification_input_size = 768
            elif args.aggregation in ['cls_start_start', 'cls_end_end', 'start_inter_start',
                                      'end_inter_end', 'start_end_start_end']:
                classification_input_size = 768 * 2
            elif args.aggregation in ['cls_start_end_start_end', 'start_end_inter_start_end']:
                classification_input_size = 768 * 3
        elif args.embed_mode == 'PubMedBERT_large':
            if args.aggregation in ['start_start', 'end_end']:
                classification_input_size = 1024
            elif args.aggregation in ['cls_start_start', 'cls_end_end', 'start_inter_start',
                                      'end_inter_end', 'start_end_start_end']:
                classification_input_size = 1024 * 2
            elif args.aggregation in ['cls_start_end_start_end', 'start_end_inter_start_end']:
                classification_input_size = 1024 * 3

        if args.exp_setting == 'binary':
            classification_output_size = 1
        elif args.exp_setting == 'multi_class':
            classification_output_size = 4

        # Bi-Linear layer?
        self.classification_layer = torch.nn.Linear(classification_input_size, classification_output_size)


    def forward(self, x, entities_range):
        x = self.tokenizer(x, return_tensors="pt",
                           padding='longest',
                           add_special_tokens=True,
                           is_split_into_words=True).to(self.device)
        x = self.model(**x)[0]

        rel_representations = []
        for i, r1 in enumerate(x):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            if self.args.aggregation == 'start_start':
                m_ent = torch.mul(r1[start_ent_1], r1[start_ent_2])
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'end_end':
                m_ent = torch.mul(r1[end_ent_1], r1[end_ent_2])
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'start_end_start_end':
                if end_ent_1 < start_ent_2:
                    m_ent_1 = torch.mul(r1[start_ent_1], r1[end_ent_1])
                    m_ent_2 = torch.mul(r1[start_ent_2], r1[end_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent_1, 0),
                                         torch.unsqueeze(m_ent_2, 0)), 1)

                else:
                    m_ent_1 = torch.mul(r1[start_ent_2], r1[end_ent_2])
                    m_ent_2 = torch.mul(r1[start_ent_1], r1[end_ent_1])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent_1, 0),
                                         torch.unsqueeze(m_ent_2, 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            if self.args.aggregation == 'cls_start_start':
                if end_ent_1 < start_ent_2:
                    m_ent = torch.mul(r1[start_ent_1], r1[start_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(m_ent, 0)), 1)
                else:
                    m_ent = torch.mul(r1[start_ent_1], r1[start_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(m_ent, 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'cls_end_end':
                if end_ent_1 < start_ent_2:
                    m_ent = torch.mul(r1[end_ent_1], r1[end_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(m_ent, 0)), 1)
                else:
                    m_ent = torch.mul(r1[end_ent_1], r1[end_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(m_ent, 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'cls_start_end_start_end':
                if end_ent_1 < start_ent_2:
                    m_ent_1 = torch.mul(r1[start_ent_1], r1[end_ent_1])
                    m_ent_2 = torch.mul(r1[start_ent_2], r1[end_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(m_ent_1, 0),
                                         torch.unsqueeze(m_ent_2, 0)), 1)
                else:
                    m_ent_1 = torch.mul(r1[start_ent_2], r1[end_ent_2])
                    m_ent_2 = torch.mul(r1[start_ent_1], r1[end_ent_1])
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(m_ent_1, 0),
                                         torch.unsqueeze(m_ent_2, 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'start_inter_start':
                if end_ent_1 < start_ent_2:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_1+1:start_ent_2], 0), 0)
                    m_ent = torch.mul(r1[start_ent_1], r1[start_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent, 0),
                                         inter_rep), 1)
                else:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_2+1:start_ent_1], 0), 0)
                    m_ent = torch.mul(r1[start_ent_1], r1[start_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent, 0),
                                         inter_rep), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'end_inter_end':
                if end_ent_1 < start_ent_2:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_1+1:start_ent_2], 0), 0)
                    m_ent = torch.mul(r1[end_ent_1], r1[end_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent, 0),
                                         inter_rep), 1)
                else:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_2 + 1:start_ent_1], 0), 0)
                    m_ent = torch.mul(r1[end_ent_1], r1[end_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent, 0),
                                         inter_rep), 1)

                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'start_end_inter_start_end':
                if end_ent_1 < start_ent_2:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_1+1:start_ent_2], 0), 0)
                    m_ent_1 = torch.mul(r1[start_ent_1], r1[end_ent_1])
                    m_ent_2 = torch.mul(r1[start_ent_2], r1[end_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent_1, 0),
                                         torch.unsqueeze(m_ent_2, 0),
                                         inter_rep), 1)
                else:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_2+1:start_ent_1], 0), 0)
                    m_ent_1 = torch.mul(r1[start_ent_2], r1[end_ent_2])
                    m_ent_2 = torch.mul(r1[start_ent_1], r1[end_ent_1])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent_1, 0),
                                         torch.unsqueeze(m_ent_2, 0),
                                         inter_rep), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))

        rel_representations_tensor = torch.stack(rel_representations, 0)

        if self.args.do_train:
            rel_representations_tensor = self.dropout(rel_representations_tensor)

        y = self.classification_layer(rel_representations_tensor)

        return y


class LMCE_bilinear(torch.nn.Module):
    def __init__(self, args, device):
        super(LMCE_bilinear, self).__init__()

        self.args = args
        self.device = device
        if args.embed_mode == 'PubMedBERT_base':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            # Add the special tokens
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
            self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
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
        elif args.embed_mode == 'PubMedBERT_large':
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

        self.dropout = torch.nn.Dropout(args.dropout)
        if args.embed_mode == 'PubMedBERT_base':
            bilinear_size = 768
            if args.aggregation in ['start_start', 'end_end']:
                classification_input_size = 768
            elif args.aggregation in ['cls_start_start', 'cls_end_end', 'start_inter_start',
                                      'end_inter_end', 'start_end_start_end']:
                classification_input_size = 768 * 2
            elif args.aggregation in ['cls_start_end_start_end', 'start_end_inter_start_end']:
                classification_input_size = 768 * 3
        elif args.embed_mode == 'PubMedBERT_large':
            bilinear_size = 1024
            if args.aggregation in ['start_start', 'end_end']:
                classification_input_size = 1024
            elif args.aggregation in ['cls_start_start', 'cls_end_end', 'start_inter_start',
                                      'end_inter_end', 'start_end_start_end']:
                classification_input_size = 1024 * 2
            elif args.aggregation in ['cls_start_end_start_end', 'start_end_inter_start_end']:
                classification_input_size = 1024 * 3

        if args.exp_setting == 'binary':
            classification_output_size = 1
        elif args.exp_setting == 'multi_class':
            classification_output_size = 4

        self.bilinear_layer_1 = torch.nn.Bilinear(bilinear_size, bilinear_size, bilinear_size)
        self.classification_layer = torch.nn.Linear(classification_input_size, classification_output_size)


    def forward(self, x, entities_range):
        x = self.tokenizer(x, return_tensors="pt",
                           padding='longest',
                           add_special_tokens=True,
                           is_split_into_words=True).to(self.device)
        x = self.model(**x)[0]

        rel_representations = []
        for i, r1 in enumerate(x):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            if self.args.aggregation == 'start_start':
                m_ent = self.bilinear_layer_1(r1[start_ent_1], r1[start_ent_2])
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'end_end':
                m_ent = self.bilinear_layer_1(r1[end_ent_1], r1[end_ent_2])
                rel_representations.append(m_ent)
            elif self.args.aggregation == 'start_end_start_end':
                if end_ent_1 < start_ent_2:
                    m_ent_1 = self.bilinear_layer_1(r1[start_ent_1], r1[end_ent_1])
                    m_ent_2 = self.bilinear_layer_1(r1[start_ent_2], r1[end_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent_1, 0),
                                         torch.unsqueeze(m_ent_2, 0)), 1)

                else:
                    m_ent_1 = self.bilinear_layer_1(r1[start_ent_2], r1[end_ent_2])
                    m_ent_2 = self.bilinear_layer_1(r1[start_ent_1], r1[end_ent_1])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent_1, 0),
                                         torch.unsqueeze(m_ent_2, 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            if self.args.aggregation == 'cls_start_start':
                if end_ent_1 < start_ent_2:
                    m_ent = self.bilinear_layer_1(r1[start_ent_1], r1[start_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(m_ent, 0)), 1)
                else:
                    m_ent = self.bilinear_layer_1(r1[start_ent_1], r1[start_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(m_ent, 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'cls_end_end':
                if end_ent_1 < start_ent_2:
                    m_ent = self.bilinear_layer_1(r1[end_ent_1], r1[end_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(m_ent, 0)), 1)
                else:
                    m_ent = self.bilinear_layer_1(r1[end_ent_1], r1[end_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(m_ent, 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'cls_start_end_start_end':
                if end_ent_1 < start_ent_2:
                    m_ent_1 = self.bilinear_layer_1(r1[start_ent_1], r1[end_ent_1])
                    m_ent_2 = self.bilinear_layer_1(r1[start_ent_2], r1[end_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(m_ent_1, 0),
                                         torch.unsqueeze(m_ent_2, 0)), 1)
                else:
                    m_ent_1 = self.bilinear_layer_1(r1[start_ent_2], r1[end_ent_2])
                    m_ent_2 = self.bilinear_layer_1(r1[start_ent_1], r1[end_ent_1])
                    rel_rep = torch.cat((torch.unsqueeze(r1[0], 0),
                                         torch.unsqueeze(m_ent_1, 0),
                                         torch.unsqueeze(m_ent_2, 0)), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'start_inter_start':
                if end_ent_1 < start_ent_2:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_1+1:start_ent_2], 0), 0)
                    m_ent = self.bilinear_layer_1(r1[start_ent_1], r1[start_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent, 0),
                                         inter_rep), 1)
                else:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_2+1:start_ent_1], 0), 0)
                    m_ent = self.bilinear_layer_1(r1[start_ent_1], r1[start_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent, 0),
                                         inter_rep), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'end_inter_end':
                if end_ent_1 < start_ent_2:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_1+1:start_ent_2], 0), 0)
                    m_ent = self.bilinear_layer_1(r1[end_ent_1], r1[end_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent, 0),
                                         inter_rep), 1)
                else:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_2 + 1:start_ent_1], 0), 0)
                    m_ent = self.bilinear_layer_1(r1[end_ent_1], r1[end_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent, 0),
                                         inter_rep), 1)

                rel_representations.append(torch.squeeze(rel_rep, 0))
            elif self.args.aggregation == 'start_end_inter_start_end':
                if end_ent_1 < start_ent_2:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_1+1:start_ent_2], 0), 0)
                    m_ent_1 = self.bilinear_layer_1(r1[start_ent_1], r1[end_ent_1])
                    m_ent_2 = self.bilinear_layer_1(r1[start_ent_2], r1[end_ent_2])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent_1, 0),
                                         torch.unsqueeze(m_ent_2, 0),
                                         inter_rep), 1)
                else:
                    inter_rep = torch.unsqueeze(torch.mean(r1[end_ent_2+1:start_ent_1], 0), 0)
                    m_ent_1 = self.bilinear_layer_1(r1[start_ent_2], r1[end_ent_2])
                    m_ent_2 = self.bilinear_layer_1(r1[start_ent_1], r1[end_ent_1])
                    rel_rep = torch.cat((torch.unsqueeze(m_ent_1, 0),
                                         torch.unsqueeze(m_ent_2, 0),
                                         inter_rep), 1)
                rel_representations.append(torch.squeeze(rel_rep, 0))

        rel_representations_tensor = torch.stack(rel_representations, 0)

        if self.args.do_train:
            rel_representations_tensor = self.dropout(rel_representations_tensor)

        y = self.classification_layer(rel_representations_tensor)

        return y
