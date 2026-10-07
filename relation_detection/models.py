"""The LaMReDA and LaMReDM relation detection models."""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.training_utils import (
    atlop_context_vector,
    inter_representation,
    load_backbone_with_markers,
)


class LaMReDA(torch.nn.Module):
    """Relation detection with additive aggregation of the representations.

    The representations chosen by ``args.aggregation`` are projected, summed and
    classified.

    Args:
        args: The experiment arguments.
        device: The device of the model inputs.
    """

    def __init__(self, args: argparse.Namespace, device: torch.device | str) -> None:
        super().__init__()

        self.args = args
        self.device = device
        self.tokenizer, self.model, classification_input_size = (
            load_backbone_with_markers(args.embed_mode)
        )

        self.dropout = torch.nn.Dropout(args.dropout)

        if args.exp_setting == "binary":
            classification_output_size = 1
        elif args.exp_setting == "multi_class":
            classification_output_size = 4

        if args.projection_dimension == 0:
            self.BN = torch.nn.BatchNorm1d(classification_input_size)
            self.head_projector = torch.nn.Linear(
                classification_input_size, classification_input_size
            )
            self.tail_projector = torch.nn.Linear(
                classification_input_size, classification_input_size
            )
            self.head_tail_projector = torch.nn.Linear(
                classification_input_size, classification_input_size
            )
            self.classification_layer = torch.nn.Linear(
                classification_input_size, classification_output_size
            )
        else:
            self.BN = torch.nn.BatchNorm1d(args.projection_dimension)
            self.head_projector = torch.nn.Linear(
                classification_input_size, args.projection_dimension
            )
            self.tail_projector = torch.nn.Linear(
                classification_input_size, args.projection_dimension
            )
            self.head_tail_projector = torch.nn.Linear(
                classification_input_size, args.projection_dimension
            )
            self.classification_layer = torch.nn.Linear(
                args.projection_dimension, classification_output_size
            )

    def forward(
        self, x: list[list[str]], entities_range: list[list[list[int]]]
    ) -> torch.Tensor:
        """Return the relation logits of a batch.

        Args:
            x: The words of each sentence, with the entity markers.
            entities_range: For each sentence, the sub-word positions of the [ent] and
                [/ent] markers of its two entities.
        """
        inputs = self.tokenizer(
            x,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=True,
            is_split_into_words=True,
        ).to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        # x = self.model(**x)[0]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
            output_hidden_states=True,
        )

        rel_representations = []
        # for i, r1 in enumerate(x):
        for i, r1 in enumerate(outputs["last_hidden_state"]):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            if self.args.aggregation == "inter":
                inter_rep = inter_representation(
                    r1, start_ent_1, end_ent_1, start_ent_2, end_ent_2
                )
                final_rep = self.head_tail_projector(inter_rep)
                rel_representations.append(final_rep)
            elif self.args.aggregation == "start_start":
                final_rep = (
                    self.head_projector(r1[start_ent_1])
                    + self.tail_projector(r1[start_ent_2])
                    + self.head_projector(r1[start_ent_2])
                    + self.tail_projector(r1[start_ent_1])
                )
                rel_representations.append(final_rep)
            elif self.args.aggregation == "end_end":
                final_rep = (
                    self.head_projector(r1[end_ent_1])
                    + self.tail_projector(r1[end_ent_2])
                    + self.head_projector(r1[end_ent_2])
                    + self.tail_projector(r1[end_ent_1])
                )
                rel_representations.append(final_rep)
            elif self.args.aggregation == "ent_context_ent_context":
                ent_1_rep = torch.mean(r1[start_ent_1 + 1 : end_ent_1], 0)
                ent_2_rep = torch.mean(r1[start_ent_2 + 1 : end_ent_2], 0)
                final_rep = (
                    self.head_projector(ent_1_rep)
                    + self.tail_projector(ent_2_rep)
                    + self.head_projector(ent_2_rep)
                    + self.tail_projector(ent_1_rep)
                )
                rel_representations.append(final_rep)
            elif self.args.aggregation == "start_end_start_end":
                final_rep = (
                    self.head_projector(r1[start_ent_1])
                    + self.tail_projector(r1[start_ent_2])
                    + self.head_projector(r1[start_ent_2])
                    + self.tail_projector(r1[start_ent_1])
                    + self.head_projector(r1[end_ent_1])
                    + self.tail_projector(r1[end_ent_2])
                    + self.head_projector(r1[end_ent_2])
                    + self.tail_projector(r1[end_ent_1])
                )
                rel_representations.append(final_rep)
            elif self.args.aggregation == "cls_start_start":
                final_rep = (
                    self.head_projector(r1[start_ent_1])
                    + self.tail_projector(r1[start_ent_2])
                    + self.head_projector(r1[start_ent_2])
                    + self.tail_projector(r1[start_ent_1])
                    + self.head_tail_projector(r1[0])
                )
                rel_representations.append(final_rep)
            elif self.args.aggregation == "cls_end_end":
                final_rep = (
                    self.head_projector(r1[end_ent_1])
                    + self.tail_projector(r1[end_ent_2])
                    + self.head_projector(r1[end_ent_2])
                    + self.tail_projector(r1[end_ent_1])
                    + self.head_tail_projector(r1[0])
                )
                rel_representations.append(final_rep)
            elif self.args.aggregation == "cls_ent_context_ent_context":
                ent_1_rep = torch.mean(r1[start_ent_1 + 1 : end_ent_1], 0)
                ent_2_rep = torch.mean(r1[start_ent_2 + 1 : end_ent_2], 0)
                final_rep = (
                    self.head_projector(ent_1_rep)
                    + self.tail_projector(ent_2_rep)
                    + self.head_projector(ent_2_rep)
                    + self.tail_projector(ent_1_rep)
                    + self.head_tail_projector(r1[0])
                )
                rel_representations.append(final_rep)
            elif self.args.aggregation == "cls_inter":
                inter_rep = inter_representation(
                    r1, start_ent_1, end_ent_1, start_ent_2, end_ent_2
                )
                final_rep = self.head_tail_projector(
                    inter_rep
                ) + self.head_tail_projector(r1[0])
                rel_representations.append(final_rep)
            elif self.args.aggregation == "cls_start_end_start_end":
                final_rep = (
                    self.head_projector(r1[start_ent_1])
                    + self.tail_projector(r1[start_ent_2])
                    + self.head_projector(r1[start_ent_2])
                    + self.tail_projector(r1[start_ent_1])
                    + self.head_projector(r1[end_ent_1])
                    + self.tail_projector(r1[end_ent_2])
                    + self.head_projector(r1[end_ent_2])
                    + self.tail_projector(r1[end_ent_1])
                    + +self.head_tail_projector(r1[0])
                )
                rel_representations.append(final_rep)
            elif self.args.aggregation == "start_inter_start":
                inter_rep = inter_representation(
                    r1, start_ent_1, end_ent_1, start_ent_2, end_ent_2
                )
                final_rep = (
                    self.head_projector(r1[start_ent_1])
                    + self.tail_projector(r1[start_ent_2])
                    + self.head_projector(r1[start_ent_2])
                    + self.tail_projector(r1[start_ent_1])
                    + self.head_tail_projector(inter_rep)
                )
                rel_representations.append(final_rep)
            elif self.args.aggregation == "end_inter_end":
                inter_rep = inter_representation(
                    r1, start_ent_1, end_ent_1, start_ent_2, end_ent_2
                )
                final_rep = (
                    self.head_projector(r1[end_ent_1])
                    + self.tail_projector(r1[end_ent_2])
                    + self.head_projector(r1[end_ent_2])
                    + self.tail_projector(r1[end_ent_1])
                    + self.head_tail_projector(inter_rep)
                )
                rel_representations.append(final_rep)
            elif self.args.aggregation == "start_end_inter_start_end":
                inter_rep = inter_representation(
                    r1, start_ent_1, end_ent_1, start_ent_2, end_ent_2
                )
                final_rep = (
                    self.head_projector(r1[start_ent_1])
                    + self.tail_projector(r1[start_ent_2])
                    + self.head_projector(r1[start_ent_2])
                    + self.tail_projector(r1[start_ent_1])
                    + self.head_projector(r1[end_ent_1])
                    + self.tail_projector(r1[end_ent_2])
                    + self.head_projector(r1[end_ent_2])
                    + self.tail_projector(r1[end_ent_1])
                    + self.head_tail_projector(inter_rep)
                )
                rel_representations.append(final_rep)
            elif self.args.aggregation == "ent_context_inter_ent_context":
                inter_rep = inter_representation(
                    r1, start_ent_1, end_ent_1, start_ent_2, end_ent_2
                )
                ent_1_rep = torch.mean(r1[start_ent_1 + 1 : end_ent_1], 0)
                ent_2_rep = torch.mean(r1[start_ent_2 + 1 : end_ent_2], 0)
                final_rep = (
                    self.head_projector(ent_1_rep)
                    + self.tail_projector(ent_2_rep)
                    + self.head_projector(ent_2_rep)
                    + self.tail_projector(ent_1_rep)
                    + self.head_tail_projector(inter_rep)
                )
                rel_representations.append(final_rep)
            elif self.args.aggregation == "atlop_context_vector_only":
                # Attention of the last layer from the [ent] markers of the two entities
                head_tail_context_vector = atlop_context_vector(
                    outputs["attentions"][-1][i],
                    r1,
                    (start_ent_1, start_ent_1),
                    (start_ent_2, start_ent_2),
                )
                final_rep = self.head_tail_projector(head_tail_context_vector)
                rel_representations.append(final_rep)
            elif self.args.aggregation == "atlop_context_vector":
                # Attention of the last layer from the [ent] markers of the two entities
                head_tail_context_vector = atlop_context_vector(
                    outputs["attentions"][-1][i],
                    r1,
                    (start_ent_1, start_ent_1),
                    (start_ent_2, start_ent_2),
                )
                final_rep = (
                    self.head_projector(r1[start_ent_1])
                    + self.tail_projector(r1[start_ent_2])
                    + self.head_projector(r1[start_ent_2])
                    + self.tail_projector(r1[start_ent_1])
                    + self.head_tail_projector(head_tail_context_vector)
                )
                rel_representations.append(final_rep)

        rel_representations_tensor = torch.stack(rel_representations, 0)

        if self.args.do_train:
            rel_representations_tensor = self.dropout(rel_representations_tensor)

        # BatchNorm needs more than one example only while training; in evaluation it
        # uses the running statistics
        if rel_representations_tensor.shape[0] != 1 or not self.training:
            y = self.BN(rel_representations_tensor)
            y = self.classification_layer(y)
        else:
            y = self.classification_layer(rel_representations_tensor)

        return y


class LaMReDM(torch.nn.Module):
    """Relation detection with multiplicative aggregation of the representations.

    The entity representations chosen by ``args.aggregation`` are projected and
    multiplied element-wise, then classified.

    Args:
        args: The experiment arguments.
        device: The device of the model inputs.
    """

    def __init__(self, args: argparse.Namespace, device: torch.device | str) -> None:
        super().__init__()

        self.args = args
        self.device = device
        self.tokenizer, self.model, classification_input_size = (
            load_backbone_with_markers(args.embed_mode)
        )

        self.dropout = torch.nn.Dropout(args.dropout)

        if args.exp_setting == "binary":
            classification_output_size = 1
        elif args.exp_setting == "multi_class":
            classification_output_size = 4

        if args.projection_dimension == 0:
            self.BN = torch.nn.BatchNorm1d(classification_input_size)
            self.head_projector = torch.nn.Linear(
                classification_input_size, classification_input_size
            )
            self.tail_projector = torch.nn.Linear(
                classification_input_size, classification_input_size
            )
            self.head_tail_projector = torch.nn.Linear(
                classification_input_size, classification_input_size
            )
            self.classification_layer = torch.nn.Linear(
                classification_input_size, classification_output_size
            )
        else:
            self.BN = torch.nn.BatchNorm1d(args.projection_dimension)
            self.head_projector = torch.nn.Linear(
                classification_input_size, args.projection_dimension
            )
            self.tail_projector = torch.nn.Linear(
                classification_input_size, args.projection_dimension
            )
            self.head_tail_projector = torch.nn.Linear(
                classification_input_size, args.projection_dimension
            )
            self.classification_layer = torch.nn.Linear(
                args.projection_dimension, classification_output_size
            )

    def forward(
        self, x: list[list[str]], entities_range: list[list[list[int]]]
    ) -> torch.Tensor:
        """Return the relation logits of a batch.

        Args:
            x: The words of each sentence, with the entity markers.
            entities_range: For each sentence, the sub-word positions of the [ent] and
                [/ent] markers of its two entities.
        """
        inputs = self.tokenizer(
            x,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=True,
            is_split_into_words=True,
        ).to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        # x = self.model(**x)[0]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
            output_hidden_states=True,
        )

        rel_representations = []
        # for i, r1 in enumerate(x):
        for i, r1 in enumerate(outputs["last_hidden_state"]):
            start_ent_1 = entities_range[i][0][0]
            end_ent_1 = entities_range[i][0][1]
            start_ent_2 = entities_range[i][1][0]
            end_ent_2 = entities_range[i][1][1]
            if self.args.aggregation == "start_start":
                m_ent_1 = self.head_projector(r1[start_ent_1]) + self.tail_projector(
                    r1[start_ent_1]
                )
                m_ent_2 = self.head_projector(r1[start_ent_2]) + self.tail_projector(
                    r1[start_ent_2]
                )
                m_ent = torch.mul(m_ent_1, m_ent_2)
                rel_representations.append(m_ent)
            elif self.args.aggregation == "end_end":
                m_ent_1 = self.head_projector(r1[end_ent_1]) + self.tail_projector(
                    r1[end_ent_1]
                )
                m_ent_2 = self.head_projector(r1[end_ent_2]) + self.tail_projector(
                    r1[end_ent_2]
                )
                m_ent = torch.mul(m_ent_1, m_ent_2)
                rel_representations.append(m_ent)
            elif self.args.aggregation == "start_end_start_end":
                m_ent_1 = torch.mul(
                    self.head_projector(r1[start_ent_1])
                    + self.tail_projector(r1[start_ent_1]),
                    self.head_projector(r1[end_ent_1])
                    + self.tail_projector(r1[end_ent_1]),
                )
                m_ent_2 = torch.mul(
                    self.head_projector(r1[start_ent_2])
                    + self.tail_projector(r1[start_ent_2]),
                    self.head_projector(r1[end_ent_2])
                    + self.tail_projector(r1[end_ent_2]),
                )
                m_ent = torch.mul(m_ent_1, m_ent_2)
                rel_representations.append(m_ent)
            if self.args.aggregation == "cls_start_start":
                m_ent_1 = self.head_projector(r1[start_ent_1]) + self.tail_projector(
                    r1[start_ent_1]
                )
                m_ent_2 = self.head_projector(r1[start_ent_2]) + self.tail_projector(
                    r1[start_ent_2]
                )
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(r1[0]))
                rel_representations.append(m_ent)
            elif self.args.aggregation == "cls_end_end":
                m_ent_1 = self.head_projector(r1[end_ent_1]) + self.tail_projector(
                    r1[end_ent_1]
                )
                m_ent_2 = self.head_projector(r1[end_ent_2]) + self.tail_projector(
                    r1[end_ent_2]
                )
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(r1[0]))
                rel_representations.append(m_ent)
            elif self.args.aggregation == "cls_start_end_start_end":
                m_ent_1 = torch.mul(
                    self.head_projector(r1[start_ent_1])
                    + self.tail_projector(r1[start_ent_1]),
                    self.head_projector(r1[end_ent_1])
                    + self.tail_projector(r1[end_ent_1]),
                )
                m_ent_2 = torch.mul(
                    self.head_projector(r1[start_ent_2])
                    + self.tail_projector(r1[start_ent_2]),
                    self.head_projector(r1[end_ent_2])
                    + self.tail_projector(r1[end_ent_2]),
                )
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(r1[0]))
                rel_representations.append(m_ent)
            elif self.args.aggregation == "start_inter_start":
                inter_rep = inter_representation(
                    r1, start_ent_1, end_ent_1, start_ent_2, end_ent_2
                )
                m_ent_1 = self.head_projector(r1[start_ent_1]) + self.tail_projector(
                    r1[start_ent_1]
                )
                m_ent_2 = self.head_projector(r1[start_ent_2]) + self.tail_projector(
                    r1[start_ent_2]
                )
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(inter_rep))
                rel_representations.append(m_ent)
            elif self.args.aggregation == "end_inter_end":
                inter_rep = inter_representation(
                    r1, start_ent_1, end_ent_1, start_ent_2, end_ent_2
                )
                m_ent_1 = self.head_projector(r1[end_ent_1]) + self.tail_projector(
                    r1[end_ent_1]
                )
                m_ent_2 = self.head_projector(r1[end_ent_2]) + self.tail_projector(
                    r1[end_ent_2]
                )
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(inter_rep))
                rel_representations.append(m_ent)
            elif self.args.aggregation == "start_end_inter_start_end":
                inter_rep = inter_representation(
                    r1, start_ent_1, end_ent_1, start_ent_2, end_ent_2
                )
                m_ent_1 = torch.mul(
                    self.head_projector(r1[start_ent_1])
                    + self.tail_projector(r1[start_ent_1]),
                    self.head_projector(r1[end_ent_1])
                    + self.tail_projector(r1[end_ent_1]),
                )
                m_ent_2 = torch.mul(
                    self.head_projector(r1[start_ent_2])
                    + self.tail_projector(r1[start_ent_2]),
                    self.head_projector(r1[end_ent_2])
                    + self.tail_projector(r1[end_ent_2]),
                )
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(inter_rep))
                rel_representations.append(m_ent)
            elif self.args.aggregation == "cls_inter":
                inter_rep = inter_representation(
                    r1, start_ent_1, end_ent_1, start_ent_2, end_ent_2
                )
                m_ent = torch.mul(
                    self.head_tail_projector(r1[0]), self.head_tail_projector(inter_rep)
                )
                rel_representations.append(m_ent)
            elif self.args.aggregation == "ent_context_ent_context":
                m_ent_1 = torch.mean(r1[start_ent_1 + 1 : end_ent_1], 0)
                m_ent_2 = torch.mean(r1[start_ent_2 + 1 : end_ent_2], 0)
                m_ent_1 = self.head_projector(m_ent_1) + self.tail_projector(m_ent_1)
                m_ent_2 = self.head_projector(m_ent_2) + self.tail_projector(m_ent_2)
                m_ent = torch.mul(m_ent_1, m_ent_2)
                rel_representations.append(m_ent)
            elif self.args.aggregation == "cls_ent_context_ent_context":
                m_ent_1 = torch.mean(r1[start_ent_1 + 1 : end_ent_1], 0)
                m_ent_2 = torch.mean(r1[start_ent_2 + 1 : end_ent_2], 0)
                m_ent_1 = self.head_projector(m_ent_1) + self.tail_projector(m_ent_1)
                m_ent_2 = self.head_projector(m_ent_2) + self.tail_projector(m_ent_2)
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(r1[0]))
                rel_representations.append(m_ent)
            elif self.args.aggregation == "ent_context_inter_ent_context":
                inter_rep = inter_representation(
                    r1, start_ent_1, end_ent_1, start_ent_2, end_ent_2
                )
                m_ent_1 = torch.mean(r1[start_ent_1 + 1 : end_ent_1], 0)
                m_ent_2 = torch.mean(r1[start_ent_2 + 1 : end_ent_2], 0)
                m_ent_1 = self.head_projector(m_ent_1) + self.tail_projector(m_ent_1)
                m_ent_2 = self.head_projector(m_ent_2) + self.tail_projector(m_ent_2)
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(m_ent, self.head_tail_projector(inter_rep))
                rel_representations.append(m_ent)
            elif self.args.aggregation == "atlop_context_vector":
                # Attention of the last layer from the [ent] markers of the two entities
                head_tail_context_vector = atlop_context_vector(
                    outputs["attentions"][-1][i],
                    r1,
                    (start_ent_1, start_ent_1),
                    (start_ent_2, start_ent_2),
                )
                m_ent_1 = torch.mean(r1[start_ent_1 + 1 : end_ent_1], 0)
                m_ent_2 = torch.mean(r1[start_ent_2 + 1 : end_ent_2], 0)
                m_ent_1 = self.head_projector(m_ent_1) + self.tail_projector(m_ent_1)
                m_ent_2 = self.head_projector(m_ent_2) + self.tail_projector(m_ent_2)
                m_ent = torch.mul(m_ent_1, m_ent_2)
                m_ent = torch.mul(
                    m_ent, self.head_tail_projector(head_tail_context_vector)
                )
                rel_representations.append(m_ent)

        rel_representations_tensor = torch.stack(rel_representations, 0)

        if self.args.do_train:
            rel_representations_tensor = self.dropout(rel_representations_tensor)

        # BatchNorm needs more than one example only while training; in evaluation it
        # uses the running statistics
        if rel_representations_tensor.shape[0] != 1 or not self.training:
            y = self.BN(rel_representations_tensor)
            y = self.classification_layer(y)
        else:
            y = self.classification_layer(rel_representations_tensor)

        return y
