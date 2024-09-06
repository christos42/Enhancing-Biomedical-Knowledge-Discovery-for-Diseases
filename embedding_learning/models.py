import sys
import torch
from transformers import AutoTokenizer, AutoModel

class LaMEL(torch.nn.Module):
    def __init__(self, args, device):
        super(LaMEL, self).__init__()

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
            if self.args.aggregation == 'start_end_start_end':
                linear_input_size = 768 * 2
            else:
                linear_input_size = 768
        elif args.embed_mode in ['BiomedBERT_large', 'BioLinkBERT_large', 'BioGPT_base', 'BioGPT_large']:
            if self.args.aggregation == 'start_end_start_end':
                linear_input_size = 1024 * 2
            else:
                linear_input_size = 1024

        self.head_projector = torch.nn.Linear(linear_input_size, linear_input_size)
        self.tail_projector= torch.nn.Linear(linear_input_size, linear_input_size)


    def forward(self, x, entities_range):
        x = self.tokenizer(x, return_tensors="pt",
                           padding='longest',
                           add_special_tokens=True,
                           is_split_into_words=True).to(self.device)
        x = self.model(**x)[0]

        ent_1_representations, ent_2_representations = [], []
        for i, r1 in enumerate(x):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            if self.args.aggregation == 'ent_context_ent_context':
                #ent_rep_1 = torch.unsqueeze(torch.mean(r1[start_ent_1 + 1:end_ent_1], 0), 0)
                ent_rep_1 = torch.mean(r1[start_ent_1 + 1:end_ent_1], 0)
                #ent_rep_2 = torch.unsqueeze(torch.mean(r1[start_ent_2 + 1:end_ent_2], 0), 0)
                ent_rep_2 = torch.mean(r1[start_ent_2 + 1:end_ent_2], 0)

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = self.head_projector(ent_rep_1) + self.tail_projector(ent_rep_1)
                ent_rep_2 = self.head_projector(ent_rep_2) + self.tail_projector(ent_rep_2)

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)
            elif self.args.aggregation == 'start_start':
                ent_rep_1 = r1[start_ent_1]
                ent_rep_2 = r1[start_ent_2]

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = self.head_projector(ent_rep_1) + self.tail_projector(ent_rep_1)
                ent_rep_2 = self.head_projector(ent_rep_2) + self.tail_projector(ent_rep_2)

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)
            elif self.args.aggregation == 'end_end':
                ent_rep_1 = r1[end_ent_1]
                ent_rep_2 = r1[end_ent_2]

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = self.head_projector(ent_rep_1) + self.tail_projector(ent_rep_1)
                ent_rep_2 = self.head_projector(ent_rep_2) + self.tail_projector(ent_rep_2)

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)
            elif self.args.aggregation == 'start_end_start_end':
                ent_rep_1 = torch.cat((r1[start_ent_1], r1[end_ent_1]), 0)
                ent_rep_2 = torch.cat((r1[start_ent_2], r1[end_ent_2]), 0)

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = self.head_projector(ent_rep_1) + self.tail_projector(ent_rep_1)
                ent_rep_2 = self.head_projector(ent_rep_2) + self.tail_projector(ent_rep_2)

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)

        ent_1_representations_tensor = torch.stack(ent_1_representations, 0)
        ent_2_representations_tensor = torch.stack(ent_2_representations, 0)

        return ent_1_representations_tensor, ent_2_representations_tensor


class LaMEL_inter(torch.nn.Module):
    def __init__(self, args, device):
        super(LaMEL_inter, self).__init__()

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
            inter_input_size = 768
            linear_input_size = 768
        elif args.embed_mode in ['BiomedBERT_large', 'BioLinkBERT_large', 'BioGPT_base', 'BioGPT_large']:
            inter_input_size = 1024
            linear_input_size = 1024

        self.head_projector = torch.nn.Linear(linear_input_size, linear_input_size)
        self.tail_projector= torch.nn.Linear(linear_input_size, linear_input_size)
        self.head_tail_projector = torch.nn.Linear(inter_input_size, inter_input_size)


    def forward(self, x, entities_range):
        x = self.tokenizer(x, return_tensors="pt",
                           padding='longest',
                           add_special_tokens=True,
                           is_split_into_words=True).to(self.device)
        x = self.model(**x)[0]

        ent_1_representations, ent_2_representations = [], []
        for i, r1 in enumerate(x):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
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
            if self.args.aggregation == 'ent_context_ent_context':
                #ent_rep_1 = torch.unsqueeze(torch.mean(r1[start_ent_1 + 1:end_ent_1], 0), 0)
                ent_rep_1 = torch.mean(r1[start_ent_1 + 1:end_ent_1], 0)
                #ent_rep_2 = torch.unsqueeze(torch.mean(r1[start_ent_2 + 1:end_ent_2], 0), 0)
                ent_rep_2 = torch.mean(r1[start_ent_2 + 1:end_ent_2], 0)

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = torch.mul((self.head_projector(ent_rep_1) + self.tail_projector(ent_rep_1)), self.head_tail_projector(inter_rep))
                ent_rep_2 = torch.mul((self.head_projector(ent_rep_2) + self.tail_projector(ent_rep_2)), self.head_tail_projector(inter_rep))

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)
            elif self.args.aggregation == 'start_start':
                ent_rep_1 = r1[start_ent_1]
                ent_rep_2 = r1[start_ent_2]

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = torch.mul((self.head_projector(ent_rep_1) + self.tail_projector(ent_rep_1)),
                                      self.head_tail_projector(inter_rep))
                ent_rep_2 = torch.mul((self.head_projector(ent_rep_2) + self.tail_projector(ent_rep_2)),
                                      self.head_tail_projector(inter_rep))

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)
            elif self.args.aggregation == 'end_end':
                ent_rep_1 = r1[end_ent_1]
                ent_rep_2 = r1[end_ent_2]

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = torch.mul((self.head_projector(ent_rep_1) + self.tail_projector(ent_rep_1)),
                                      self.head_tail_projector(inter_rep))
                ent_rep_2 = torch.mul((self.head_projector(ent_rep_2) + self.tail_projector(ent_rep_2)),
                                      self.head_tail_projector(inter_rep))

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)
            elif self.args.aggregation == 'start_end_start_end':
                ent_rep_1 = torch.mul(r1[start_ent_1], r1[end_ent_1])
                ent_rep_2 = torch.mul(r1[start_ent_2], r1[end_ent_2])

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = torch.mul((self.head_projector(ent_rep_1) + self.tail_projector(ent_rep_1)),
                                      self.head_tail_projector(inter_rep))
                ent_rep_2 = torch.mul((self.head_projector(ent_rep_2) + self.tail_projector(ent_rep_2)),
                                      self.head_tail_projector(inter_rep))

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)

        ent_1_representations_tensor = torch.stack(ent_1_representations, 0)
        ent_2_representations_tensor = torch.stack(ent_2_representations, 0)

        return ent_1_representations_tensor, ent_2_representations_tensor