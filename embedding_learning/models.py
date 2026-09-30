from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.training_utils import inter_representation, load_backbone_with_markers


class LaMEL(torch.nn.Module):
    def __init__(self, args: argparse.Namespace, device: torch.device | str) -> None:
        super().__init__()

        self.args = args
        self.device = device
        self.tokenizer, self.model, hidden_size = load_backbone_with_markers(
            args.embed_mode
        )

        self.dropout = torch.nn.Dropout(args.dropout)
        if self.args.aggregation == "start_end_start_end":
            linear_input_size = hidden_size * 2
        else:
            linear_input_size = hidden_size

        self.head_projector = torch.nn.Linear(linear_input_size, linear_input_size)
        self.tail_projector = torch.nn.Linear(linear_input_size, linear_input_size)

    def forward(
        self, x: list[list[str]], entities_range: list[list[list[int]]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            x,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=True,
            is_split_into_words=True,
        ).to(self.device)
        outputs = self.model(**inputs)[0]

        ent_1_representations, ent_2_representations = [], []
        for i, r1 in enumerate(outputs):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            if self.args.aggregation == "ent_context_ent_context":
                # ent_rep_1 = torch.unsqueeze(
                #     torch.mean(r1[start_ent_1 + 1:end_ent_1], 0), 0)
                ent_rep_1 = torch.mean(r1[start_ent_1 + 1 : end_ent_1], 0)
                # ent_rep_2 = torch.unsqueeze(
                #     torch.mean(r1[start_ent_2 + 1:end_ent_2], 0), 0)
                ent_rep_2 = torch.mean(r1[start_ent_2 + 1 : end_ent_2], 0)

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = self.head_projector(ent_rep_1) + self.tail_projector(
                    ent_rep_1
                )
                ent_rep_2 = self.head_projector(ent_rep_2) + self.tail_projector(
                    ent_rep_2
                )

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)
            elif self.args.aggregation == "start_start":
                ent_rep_1 = r1[start_ent_1]
                ent_rep_2 = r1[start_ent_2]

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = self.head_projector(ent_rep_1) + self.tail_projector(
                    ent_rep_1
                )
                ent_rep_2 = self.head_projector(ent_rep_2) + self.tail_projector(
                    ent_rep_2
                )

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)
            elif self.args.aggregation == "end_end":
                ent_rep_1 = r1[end_ent_1]
                ent_rep_2 = r1[end_ent_2]

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = self.head_projector(ent_rep_1) + self.tail_projector(
                    ent_rep_1
                )
                ent_rep_2 = self.head_projector(ent_rep_2) + self.tail_projector(
                    ent_rep_2
                )

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)
            elif self.args.aggregation == "start_end_start_end":
                ent_rep_1 = torch.cat((r1[start_ent_1], r1[end_ent_1]), 0)
                ent_rep_2 = torch.cat((r1[start_ent_2], r1[end_ent_2]), 0)

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = self.head_projector(ent_rep_1) + self.tail_projector(
                    ent_rep_1
                )
                ent_rep_2 = self.head_projector(ent_rep_2) + self.tail_projector(
                    ent_rep_2
                )

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)

        ent_1_representations_tensor = torch.stack(ent_1_representations, 0)
        ent_2_representations_tensor = torch.stack(ent_2_representations, 0)

        return ent_1_representations_tensor, ent_2_representations_tensor


class LaMELInter(torch.nn.Module):
    def __init__(self, args: argparse.Namespace, device: torch.device | str) -> None:
        super().__init__()

        self.args = args
        self.device = device
        self.tokenizer, self.model, hidden_size = load_backbone_with_markers(
            args.embed_mode
        )

        self.dropout = torch.nn.Dropout(args.dropout)
        inter_input_size = hidden_size
        linear_input_size = hidden_size

        self.head_projector = torch.nn.Linear(linear_input_size, linear_input_size)
        self.tail_projector = torch.nn.Linear(linear_input_size, linear_input_size)
        self.head_tail_projector = torch.nn.Linear(inter_input_size, inter_input_size)

    def forward(
        self, x: list[list[str]], entities_range: list[list[list[int]]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            x,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=True,
            is_split_into_words=True,
        ).to(self.device)
        outputs = self.model(**inputs)[0]

        ent_1_representations, ent_2_representations = [], []
        for i, r1 in enumerate(outputs):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            inter_rep = inter_representation(
                r1, start_ent_1, end_ent_1, start_ent_2, end_ent_2
            )
            if self.args.aggregation == "ent_context_ent_context":
                # ent_rep_1 = torch.unsqueeze(
                #     torch.mean(r1[start_ent_1 + 1:end_ent_1], 0), 0)
                ent_rep_1 = torch.mean(r1[start_ent_1 + 1 : end_ent_1], 0)
                # ent_rep_2 = torch.unsqueeze(
                #     torch.mean(r1[start_ent_2 + 1:end_ent_2], 0), 0)
                ent_rep_2 = torch.mean(r1[start_ent_2 + 1 : end_ent_2], 0)

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = torch.mul(
                    (self.head_projector(ent_rep_1) + self.tail_projector(ent_rep_1)),
                    self.head_tail_projector(inter_rep),
                )
                ent_rep_2 = torch.mul(
                    (self.head_projector(ent_rep_2) + self.tail_projector(ent_rep_2)),
                    self.head_tail_projector(inter_rep),
                )

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)
            elif self.args.aggregation == "start_start":
                ent_rep_1 = r1[start_ent_1]
                ent_rep_2 = r1[start_ent_2]

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = torch.mul(
                    (self.head_projector(ent_rep_1) + self.tail_projector(ent_rep_1)),
                    self.head_tail_projector(inter_rep),
                )
                ent_rep_2 = torch.mul(
                    (self.head_projector(ent_rep_2) + self.tail_projector(ent_rep_2)),
                    self.head_tail_projector(inter_rep),
                )

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)
            elif self.args.aggregation == "end_end":
                ent_rep_1 = r1[end_ent_1]
                ent_rep_2 = r1[end_ent_2]

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = torch.mul(
                    (self.head_projector(ent_rep_1) + self.tail_projector(ent_rep_1)),
                    self.head_tail_projector(inter_rep),
                )
                ent_rep_2 = torch.mul(
                    (self.head_projector(ent_rep_2) + self.tail_projector(ent_rep_2)),
                    self.head_tail_projector(inter_rep),
                )

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)
            elif self.args.aggregation == "start_end_start_end":
                ent_rep_1 = torch.mul(r1[start_ent_1], r1[end_ent_1])
                ent_rep_2 = torch.mul(r1[start_ent_2], r1[end_ent_2])

                if self.args.do_train:
                    ent_rep_1 = self.dropout(ent_rep_1)
                    ent_rep_2 = self.dropout(ent_rep_2)

                ent_rep_1 = torch.mul(
                    (self.head_projector(ent_rep_1) + self.tail_projector(ent_rep_1)),
                    self.head_tail_projector(inter_rep),
                )
                ent_rep_2 = torch.mul(
                    (self.head_projector(ent_rep_2) + self.tail_projector(ent_rep_2)),
                    self.head_tail_projector(inter_rep),
                )

                ent_1_representations.append(ent_rep_1)
                ent_2_representations.append(ent_rep_2)

        ent_1_representations_tensor = torch.stack(ent_1_representations, 0)
        ent_2_representations_tensor = torch.stack(ent_2_representations, 0)

        return ent_1_representations_tensor, ent_2_representations_tensor
