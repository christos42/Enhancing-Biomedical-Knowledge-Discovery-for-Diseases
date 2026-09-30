import argparse
import logging
import os
import sys

import torch
from sklearn.metrics import fbeta_score, precision_recall_fscore_support
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from dataloader import dataloader
from models import LMREA, LMREM, LMREAProj, LMREAttention, LMREMProj

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from utils.training_utils import SaveResults, set_seed

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def evaluate(test_batch, loss_fn, args, test_or_dev):
    steps, test_loss = 0, 0
    all_predictions, all_gold_labels = [], []

    with torch.no_grad():
        for data in test_batch:
            steps += 1
            # Inference
            text = data[0]
            entities_range = data[1]
            relation_gold = data[2]
            all_gold_labels.extend(relation_gold)

            logits = model(text, entities_range)
            if args.exp_setting == "binary":
                sigmoid = torch.nn.Sigmoid()
                predicted_probabilities = sigmoid(logits)
                predictions = (predicted_probabilities > 0.5).int()
                predictions = torch.squeeze(predictions.T, 0)
                all_predictions.extend(predictions.tolist())
                relation_gold_tensor = (
                    torch.tensor(relation_gold)
                    .view(len(relation_gold), -1)
                    .float()
                    .to(device)
                )
            elif args.exp_setting == "multi_class":
                softmax = torch.nn.Softmax(dim=1)
                predicted_probabilities = softmax(logits)
                predictions = torch.argmax(predicted_probabilities, 1)
                all_predictions.extend(predictions.tolist())
                relation_gold_tensor = torch.tensor(relation_gold).to(device)

            # Evaluation
            loss_ev = loss_fn(logits, relation_gold_tensor)
            test_loss += loss_ev

        if args.exp_setting == "binary":
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_gold_labels, all_predictions, average="binary", zero_division=0.0
            )
            f_0_5 = fbeta_score(
                all_gold_labels,
                all_predictions,
                beta=0.5,
                average="binary",
                zero_division=0.0,
            )
        elif args.exp_setting == "multi_class":
            # eval_metric is the scikit-learn averaging: micro, macro or weighted
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_gold_labels,
                all_predictions,
                average=args.eval_metric,
                zero_division=0.0,
            )
            f_0_5 = fbeta_score(
                all_gold_labels,
                all_predictions,
                beta=0.5,
                average=args.eval_metric,
                zero_division=0.0,
            )

        logger.info(f"------ {test_or_dev} Results ------")
        logger.info(f"loss : {test_loss / steps:.4f}")
        logger.info(
            f"precision={precision:.4f}, recall={recall:.4f}, "
            f"f1={f1:.4f}, f_0_5={f_0_5:.4f}"
        )

    return precision, recall, f1, f_0_5, test_loss / steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="the path to the dataset"
    )

    parser.add_argument(
        "--dataset_path_eval", type=str, help="the path to the dataset for evaluation"
    )

    parser.add_argument("--do_train", action="store_true", help="training mode")

    parser.add_argument(
        "--do_eval", action="store_true", help="whether or not to evaluate the model"
    )

    parser.add_argument(
        "--do_cross_disease_training",
        action="store_true",
        help="whether cross-disease training/evaluation is applied",
    )

    parser.add_argument(
        "--model_id",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="the model id: 1 (LMCE_proj), 2 (LMCE_mul_proj)",
    )

    parser.add_argument(
        "--epoch", default=100, type=int, help="number of training epoch"
    )

    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="number of samples in one training batch",
    )

    parser.add_argument(
        "--eval_batch_size",
        default=32,
        type=int,
        help="number of samples in one testing batch",
    )

    parser.add_argument(
        "--embed_mode",
        type=str,
        required=True,
        choices=["PubMedBERT_base", "PubMedBERT_large"],
        help="PubMedBERT_base, PubMedBERT_large",
    )

    parser.add_argument(
        "--exp_setting",
        type=str,
        required=True,
        choices=["binary", "multi_class"],
        help="the experimental setting for the task (relation detection): binary or "
        "multi_class",
    )

    parser.add_argument(
        "--eval_metric",
        type=str,
        choices=["micro", "macro", "weighted"],
        help="micro, macro or weighted f1 (weighted is what 'macro' computed before)",
    )

    parser.add_argument("--lr", default=None, type=float, help="initial learning rate")

    parser.add_argument(
        "--weight_decay", default=0, type=float, help="weight decaying rate"
    )

    parser.add_argument("--seed", default=42, type=int, help="random seed initiation")

    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="dropout rate for the input of the classification layer",
    )

    parser.add_argument(
        "--do_gradient_clipping",
        action="store_true",
        help="whether or not to do gradient clipping",
    )

    parser.add_argument(
        "--clip", default=0.5, type=float, help="the max norm of the gradient"
    )

    parser.add_argument(
        "--steps", default=50, type=int, help="show result for every 50 steps"
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="the output directory"
    )

    parser.add_argument(
        "--output_file",
        default="test",
        type=str,
        required=True,
        help="name of result file",
    )

    parser.add_argument(
        "--sentence_wise_splits",
        action="store_true",
        help="whether or not to split the dataset in a sentence-wise way",
    )

    parser.add_argument(
        "--fold",
        default=0,
        type=int,
        help="the id of the split if cross-validation is applied",
    )

    parser.add_argument(
        "--encoding_layer",
        type=int,
        required=False,
        choices=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
        ],
        help="the encoding layer of the LM to extract the representation",
    )

    parser.add_argument(
        "--attention_head",
        type=int,
        required=False,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        help="the attention head of the LM to extract the attention scores",
    )

    parser.add_argument(
        "--aggregation",
        type=str,
        default="ent_context_ent_context",
        required=False,
        choices=[
            "ent_context_ent_context",
            "atlop_context_vector",
            "atlop_context_vector_only",
            "layer_specific",
            "head_specific",
            "non_specific",
        ],
        help="the aggregation strategy after the LM",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cuda"
    set_seed(args.seed)

    output_dir = args.output_dir + args.output_file
    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.addHandler(
        logging.FileHandler(output_dir + "/" + args.output_file + ".log", "w")
    )
    logger.info(sys.argv)
    logger.info(args)

    saved_file = SaveResults(
        output_dir + "/" + args.output_file + ".txt",
        header="# epoch \t train_loss \t  dev_loss \t test_loss \t dev_precision "
        "\t dev_recall \t dev_f1 \t dev_f_0_5 \t test_precision \t test_recall "
        "\t test_f1 \t test_f_0_5",
    )

    model_file = args.output_file + ".pt"

    train_batch, test_batch, dev_batch = dataloader(args)

    if args.do_train:
        logger.info("------Training------")
        if args.model_id == 1:
            model = LMREA(args, device)
        elif args.model_id == 2:
            model = LMREAProj(args, device)
        elif args.model_id == 3:
            model = LMREM(args, device)
        elif args.model_id == 4:
            model = LMREMProj(args, device)
        elif args.model_id == 5:
            model = LMREAttention(args, device)

        model.to(device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.exp_setting == "binary":
            loss_fn = BCEWithLogitsLoss()
        elif args.exp_setting == "multi_class":
            loss_fn = CrossEntropyLoss()

        best_result = 0
        test_precision_best = None
        test_recall_best = None
        test_f1_best = None
        test_f_0_5_best = None
        # Training
        for epoch in range(args.epoch):
            steps, train_loss = 0, 0

            model.train()
            for data in tqdm(train_batch):
                steps += 1
                optimizer.zero_grad()

                text = data[0]
                entities_range = data[1]
                relation_gold = data[2]
                if args.exp_setting == "binary":
                    relation_gold_tensor = (
                        torch.tensor(relation_gold)
                        .view(len(relation_gold), -1)
                        .float()
                        .to(device)
                    )
                elif args.exp_setting == "multi_class":
                    relation_gold_tensor = torch.tensor(relation_gold).to(device)

                logits = model(text, entities_range)

                loss_t = loss_fn(logits, relation_gold_tensor)

                loss_t.backward()

                train_loss += loss_t.item()
                if args.do_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=model.parameters(), max_norm=args.clip
                    )
                optimizer.step()

                if steps % args.steps == 0:
                    logger.info(
                        f"Epoch: {epoch}, step: {steps} / {len(train_batch)}, "
                        f"loss = {train_loss / steps:.4f}"
                    )

            logger.info("------ Training Set Results ------")
            logger.info(f"loss : {train_loss / steps:.4f}")

            if args.do_eval:
                model.eval()
                logger.info("------ Testing ------")
                dev_precision, dev_recall, dev_f1, dev_f_0_5, dev_loss = evaluate(
                    dev_batch, loss_fn, args, "dev"
                )
                test_precision, test_recall, test_f1, test_f_0_5, test_loss = evaluate(
                    test_batch, loss_fn, args, "test"
                )

                if epoch == 0 or dev_f1 > best_result:
                    # if epoch == 0 or dev_f_0_5 > best_result:
                    best_result = dev_f1
                    # best_result = dev_f_0_5
                    test_precision_best = test_precision
                    test_recall_best = test_recall
                    test_f1_best = test_f1
                    test_f_0_5_best = test_f_0_5
                    # torch.save(model.state_dict(), output_dir + "/" + model_file)
                    logger.info("Best result on dev saved!!!")

                saved_file.save(
                    f"{epoch} \t {train_loss / steps:.4f} \t {dev_loss:.4f} \t "
                    f"{test_loss:.4f} \t {dev_precision:.4f} \t {dev_recall:.4f} \t "
                    f"{dev_f1:.4f} \t {dev_f_0_5:.4f} \t {test_precision:.4f} \t "
                    f"{test_recall:.4f} \t {test_f1:.4f} \t {test_f_0_5:.4f}"
                )

        saved_file.save(
            f"best test results: precision: {test_precision_best:.4f} \t "
            f"recall: {test_recall_best:.4f} \t f1: {test_f1_best:.4f}  \t "
            f"f_0_5: {test_f_0_5_best:.4f}"
        )
