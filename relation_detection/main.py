import argparse
import os
import logging
import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, fbeta_score
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from dataloader import *
from models import LaMReDA, LaMReDM


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class save_results(object):
    def __init__(self, filename, header=None):
        self.filename = filename
        if os.path.exists(filename):
            os.remove(filename)

        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)

    def save(self, info):
        with open(self.filename, 'a') as out:
            print(info, file=out)


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
            if args.exp_setting == 'binary':
                sigmoid = torch.nn.Sigmoid()
                predicted_probabilities = sigmoid(logits)
                predictions = (predicted_probabilities > 0.5).int()
                predictions = torch.squeeze(predictions.T, 0)
                all_predictions.extend(predictions.tolist())
                relation_gold_tensor = torch.tensor(relation_gold).view(len(relation_gold), -1).float().to(device)
            elif args.exp_setting == 'multi_class':
                softmax = torch.nn.Softmax(dim=1)
                predicted_probabilities = softmax(logits)
                predictions = torch.argmax(predicted_probabilities, 1)
                all_predictions.extend(predictions.tolist())
                relation_gold_tensor = torch.tensor(relation_gold).to(device)

            # Evaluation
            loss_ev = loss_fn(logits, relation_gold_tensor)
            test_loss += loss_ev

        if args.exp_setting == 'binary':
            precision, recall, f1, _ = precision_recall_fscore_support(all_gold_labels,
                                                                       all_predictions,
                                                                       average='binary',
                                                                       zero_division=0.0)
            f_0_5 = fbeta_score(all_gold_labels,
                                all_predictions,
                                beta=0.5,
                                average='binary',
                                zero_division=0.0)
        elif args.exp_setting == 'multi_class':
            if args.eval_metric == "micro":
                precision, recall, f1, _ = precision_recall_fscore_support(all_gold_labels,
                                                                           all_predictions,
                                                                           average='micro',
                                                                           zero_division=0.0)
                f_0_5 = fbeta_score(all_gold_labels,
                                    all_predictions,
                                    beta=0.5,
                                    average='micro',
                                    zero_division=0.0)
            elif args.eval_metric == 'macro':
                precision, recall, f1, _ = precision_recall_fscore_support(all_gold_labels,
                                                                           all_predictions,
                                                                           average='weighted',
                                                                           zero_division=0.0)
                f_0_5 = fbeta_score(all_gold_labels,
                                    all_predictions,
                                    beta=0.5,
                                    average='weighted',
                                    zero_division=0.0)



        logger.info("------ {} Results ------".format(test_or_dev))
        logger.info("loss : {:.4f}".format(test_loss / steps))
        logger.info("precision={:.4f}, recall={:.4f}, f1={:.4f}, f_0_5={:.4f}".format(precision, recall, f1, f_0_5))

    return precision, recall, f1, f_0_5, test_loss / steps


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str,
                        help="the path to the dataset")

    parser.add_argument("--dataset_path_eval", type=str,
                        help="the path to the dataset for evaluation")

    parser.add_argument("--dataset_path_train", type=str,
                        help="the path to the dataset for training")

    parser.add_argument("--dataset_path_dev", type=str,
                        help="the path to the dataset for development/validation")

    parser.add_argument("--dataset_path_test", type=str,
                        help="the path to the dataset for testing")

    parser.add_argument("--do_train", action="store_true",
                        help="training mode")

    parser.add_argument("--do_eval", action="store_true",
                        help="whether or not to evaluate the model")

    parser.add_argument("--do_end_to_end_training", action="store_true",
                        help="whether cross-disease training/evaluation is applied")

    parser.add_argument("--do_cross_disease_training", action="store_true",
                        help="training the model using the full dataset for a specific number of epochs")

    parser.add_argument("--model_id", type=int, choices=[1, 2],
                        help="the model id: 1 (LaMReDA), 2 (LaMReDM)")

    parser.add_argument("--epoch", default=100, type=int,
                        help="number of training epoch")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="number of samples in one training batch")

    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="number of samples in one testing batch")

    parser.add_argument("--embed_mode", default=None, type=str, required=True,
                        choices=["BiomedBERT_base", "BiomedBERT_large",
                                 "BioLinkBERT_base", "BioLink_large",
                                 "BioGPT_base", "BioGPT_large"],
                        help="BiomedBERT_base, BiomedBERT_large, BioLinkBERT_base, BioLink_large, BioGPT_base, BioGPT_large")

    parser.add_argument("--exp_setting", default="binary", type=str, required=True,
                        choices=["binary", "multi_class"],
                        help="the experimental setting for the task (relation detection): binary or multi_class")

    parser.add_argument("--eval_metric", default="micro", type=str, choices=["micro", "macro"],
                        help="micro f1 or macro f1")

    parser.add_argument("--lr", default=None, type=float,
                        help="initial learning rate")

    parser.add_argument("--weight_decay", default=0, type=float,
                        help="weight decaying rate")

    parser.add_argument("--seed", default=42, type=int,
                        help="random seed initiation")

    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate for the input of the classification layer")

    parser.add_argument("--do_gradient_clipping", action="store_true",
                        help="whether or not to do gradient clipping")

    parser.add_argument("--clip", default=0.25, type=float,
                        help="the max norm of the gradient")

    parser.add_argument("--steps", default=50, type=int,
                        help="show result for every 50 steps")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="the output directory")

    parser.add_argument("--output_file", default="test", type=str, required=True,
                        help="name of result file")

    parser.add_argument("--sentence_wise_splits", action="store_true",
                        help="whether or not to split the dataset in a sentence-wise way")

    parser.add_argument("--do_cross_validation", action="store_true",
                        help="whether or not to perform cross validation")

    parser.add_argument("--fold", default=0, type=int,
                        help="the id of the split if cross-validation is applied")

    parser.add_argument("--aggregation", type=str, required=True,
                        choices=["start_start", "end_end", "ent_context_ent_context", "start_end_start_end", "inter",
                                 "cls_start_start", "cls_end_end", "cls_ent_context_ent_context", "cls_inter",
                                 "cls_start_end_start_end", "start_inter_start", "end_inter_end", "start_end_inter_start_end",
                                 "ent_context_inter_ent_context", "atlop_context_vector_only", "atlop_context_vector"],
                        help="the aggregation strategy after the LM")

    parser.add_argument("--save_best_weights", action="store_true",
                        help="save the best model weights based on the performance on development set")

    parser.add_argument("--use_distantly_supervised_data", action="store_true",
                        help="whether or not to use the distantly supervised dataset")

    parser.add_argument("--projection_dimension", default = 0, type=int,
                        help="the dimension of the projection layer")


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cuda"
    set_seed(args.seed)

    output_dir = args.output_dir + args.output_file
    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.addHandler(logging.FileHandler(output_dir + "/" + args.output_file + ".log", 'w'))
    logger.info(sys.argv)
    logger.info(args)

    if args.do_end_to_end_training:
        saved_file = save_results(output_dir + "/" + args.output_file + ".txt",
                                  header="# epoch \t train_loss \t test_loss \t test_precision \t test_recall \t test_f1 \t test_f_0_5")
    else:
        saved_file = save_results(output_dir + "/" + args.output_file + ".txt",
                                  header="# epoch \t train_loss \t  dev_loss \t test_loss \t dev_precision \t dev_recall \t dev_f1 \t dev_f_0_5 \t test_precision \t test_recall \t test_f1 \t test_f_0_5")

    model_file = args.output_file + ".pt"

    if args.do_end_to_end_training:
        train_batch, test_batch = dataloader(args)
    else:
        train_batch, test_batch, dev_batch = dataloader(args)

    if args.do_train:
        logger.info("------Training------")
        if args.model_id == 1:
            model = LaMReDA(args, device)
        elif args.model_id == 2:
            model = LaMReDM(args, device)

        model.to(device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.exp_setting == 'binary':
            loss_fn = BCEWithLogitsLoss()
        elif args.exp_setting == 'multi_class':
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
                if args.exp_setting == 'binary':
                    relation_gold_tensor = torch.tensor(relation_gold).view(len(relation_gold), -1).float().to(device)
                elif args.exp_setting == 'multi_class':
                    relation_gold_tensor = torch.tensor(relation_gold).to(device)

                logits = model(text, entities_range)

                loss_t = loss_fn(logits, relation_gold_tensor)

                loss_t.backward()

                train_loss += loss_t.item()
                if args.do_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.clip)
                optimizer.step()

                if steps % args.steps == 0:
                    logger.info("Epoch: {}, step: {} / {}, loss = {:.4f}".format
                                (epoch, steps, len(train_batch), train_loss / steps))

            logger.info("------ Training Set Results ------")
            logger.info("loss : {:.4f}".format(train_loss / steps))

            if args.do_eval:
                model.eval()
                logger.info("------ Testing ------")
                if args.do_end_to_end_training:
                    test_precision, test_recall, test_f1, test_f_0_5, test_loss = evaluate(test_batch, loss_fn, args,
                                                                                           "test")

                    saved_file.save(
                        "{} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}".format(
                            epoch,
                            train_loss / steps,
                            test_loss,
                            test_precision,
                            test_recall,
                            test_f1,
                            test_f_0_5))
                else:
                    dev_precision, dev_recall, dev_f1, dev_f_0_5, dev_loss = evaluate(dev_batch, loss_fn, args, "dev")
                    test_precision, test_recall, test_f1, test_f_0_5, test_loss = evaluate(test_batch, loss_fn, args, "test")

                    if epoch == 0 or dev_f1 > best_result:
                    #if epoch == 0 or dev_f_0_5 > best_result:
                        best_result = dev_f1
                        #best_result = dev_f_0_5
                        test_precision_best = test_precision
                        test_recall_best = test_recall
                        test_f1_best = test_f1
                        test_f_0_5_best = test_f_0_5
                        if args.save_best_weights:
                            torch.save(model.state_dict(), output_dir + "/" + model_file)
                        logger.info("Best result on dev saved!!!")


                    saved_file.save("{} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}".format(epoch,
                                                                                                                                                              train_loss / steps,
                                                                                                                                                              dev_loss,
                                                                                                                                                              test_loss,
                                                                                                                                                              dev_precision,
                                                                                                                                                              dev_recall,
                                                                                                                                                              dev_f1,
                                                                                                                                                              dev_f_0_5,
                                                                                                                                                              test_precision,
                                                                                                                                                              test_recall,
                                                                                                                                                              test_f1,
                                                                                                                                                              test_f_0_5_best))

        if args.do_end_to_end_training:
            torch.save(model.state_dict(), output_dir + '/' + model_file)
        else:
            saved_file.save(
                "best test results: precision: {:.4f} \t recall: {:.4f} \t f1: {:.4f}  \t f_0_5: {:.4f}".format(
                    test_precision_best,
                    test_recall_best,
                    test_f1_best,
                    test_f_0_5_best))