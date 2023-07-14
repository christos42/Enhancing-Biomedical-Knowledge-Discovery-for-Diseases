import argparse
import os
import logging
import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from dataloader import *
from models import LMCE, LMCE_mul, LMCE_bilinear


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
            correlation_gold = data[2]
            all_gold_labels.extend(correlation_gold)

            logits = model(text, entities_range)
            if args.exp_setting == 'binary':
                sigmoid = torch.nn.Sigmoid()
                predicted_probabilities = sigmoid(logits)
                predictions = (predicted_probabilities > 0.5).int()
                predictions = torch.squeeze(predictions.T, 0)
                all_predictions.extend(predictions.tolist())
                correlation_gold_tensor = torch.tensor(correlation_gold).view(len(correlation_gold), -1).float().to(device)
            elif args.exp_setting == 'multi_class':
                softmax = torch.nn.Softmax(dim=1)
                predicted_probabilities = softmax(logits)
                predictions = torch.argmax(predicted_probabilities, 1)
                all_predictions.extend(predictions.tolist())
                correlation_gold_tensor = torch.tensor(correlation_gold).to(device)

            # Evaluation
            loss_ev = loss_fn(logits, correlation_gold_tensor)
            test_loss += loss_ev

        if args.exp_setting == 'binary':
            precision, recall, f1, _ = precision_recall_fscore_support(all_gold_labels,
                                                                       all_predictions,
                                                                       average='binary')
        elif args.exp_setting == 'multi_class':
            if args.eval_metric == "micro":
                precision, recall, f1, _ = precision_recall_fscore_support(all_gold_labels,
                                                                           all_predictions,
                                                                           average='micro')
            elif args.eval_metric == 'macro':
                precision, recall, f1, _ = precision_recall_fscore_support(all_gold_labels,
                                                                           all_predictions,
                                                                           average='macro')



        logger.info("------ {} Results ------".format(test_or_dev))
        logger.info("loss : {:.4f}".format(test_loss / steps))
        logger.info("precision={:.4f}, recall={:.4f}, f1={:.4f}".format(precision, recall, f1))

    return precision, recall, f1, test_loss / steps


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default='essential_dataset_total.json', type=str, required=True,
                        help="the path to the dataset")

    parser.add_argument("--dataset_path_eval", type=str,
                        help="the path to the dataset")

    parser.add_argument("--do_train", action="store_true",
                        help="training mode")

    parser.add_argument("--do_eval", action="store_true",
                        help="whether or not to evaluate the model")

    parser.add_argument("--do_cross_disease_training", action="store_true",
                        help="whether cross-disease training/evaluation is applied")

    parser.add_argument("--model_id", type=int,
                        help="the model id: 1 (LMCE), 2 (LMCE_mul)")

    parser.add_argument("--epoch", default=100, type=int,
                        help="number of training epoch")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="number of samples in one training batch")

    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="number of samples in one testing batch")

    parser.add_argument("--embed_mode", default=None, type=str, required=True,
                        help="PubMedBERT_base, PubMedBERT_large")

    parser.add_argument("--exp_setting", default="binary", type=str, required=True,
                        help="the experimental setting for the task (correlation detection): binary or multi_class")

    parser.add_argument("--eval_metric", default="micro", type=str,
                        help="micro f1 or macro f1")

    parser.add_argument("--lr", default=None, type=float,
                        help="initial learning rate")

    parser.add_argument("--weight_decay", default=0, type=float,
                        help="weight decaying rate")

    parser.add_argument("--seed", default=42, type=int,
                        help="random seed initiation")

    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate for the input of the classification layer")

    parser.add_argument("--steps", default=50, type=int,
                        help="show result for every 50 steps")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="the output directory")

    parser.add_argument("--output_file", default="test", type=str, required=True,
                        help="name of result file")

    parser.add_argument("--sentence_wise_splits", action="store_true",
                        help="whether or not to split the dataset in a sentence-wise way")

    parser.add_argument("--fold", default=0, type=int,
                        help="the id of the split if cross-validation is applied")

    parser.add_argument("--aggregation", type=str, required=True,
                        help="the aggregation strategy after the LM: 'start_start', 'end_end', 'cls_start_start', "
                             "'cls_end_end', 'start_inter_start', 'end_inter_end', 'start_end_start_end', "
                             "'cls_start_end_start_end', 'start_end_inter_start_end'")


    args = parser.parse_args()

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    set_seed(args.seed)

    output_dir = args.output_dir + args.output_file
    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.addHandler(logging.FileHandler(output_dir + "/" + args.output_file + ".log", 'w'))
    logger.info(sys.argv)
    logger.info(args)

    saved_file = save_results(output_dir + "/" + args.output_file + ".txt",
                              header="# epoch \t train_loss \t  dev_loss \t test_loss \t dev_precision \t dev_recall \t dev_f1 \t test_precision \t test_recall \t test_f1")

    model_file = args.output_file + ".pt"

    train_batch, test_batch, dev_batch = dataloader(args)

    if args.do_train:
        logger.info("------Training------")
        if args.model_id == 1:
            model = LMCE(args, device)
        elif args.model_id == 2:
            model = LMCE_mul(args, device)
        elif args.model_id == 3:
            model = LMCE_bilinear(args, device)

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
        # Training
        for epoch in range(args.epoch):
            steps, train_loss = 0, 0

            model.train()
            for data in tqdm(train_batch):
                steps += 1
                optimizer.zero_grad()

                text = data[0]
                entities_range = data[1]
                correlation_gold = data[2]
                if args.exp_setting == 'binary':
                    correlation_gold_tensor = torch.tensor(correlation_gold).view(len(correlation_gold), -1).float().to(device)
                elif args.exp_setting == 'multi_class':
                    correlation_gold_tensor = torch.tensor(correlation_gold).to(device)

                logits = model(text, entities_range)

                loss_t = loss_fn(logits, correlation_gold_tensor)

                loss_t.backward()

                train_loss += loss_t.item()
                #torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.clip)
                optimizer.step()

                if steps % args.steps == 0:
                    logger.info("Epoch: {}, step: {} / {}, loss = {:.4f}".format
                                (epoch, steps, len(train_batch), train_loss / steps))

            logger.info("------ Training Set Results ------")
            logger.info("loss : {:.4f}".format(train_loss / steps))

            if args.do_eval:
                model.eval()
                logger.info("------ Testing ------")
                dev_precision, dev_recall, dev_f1, dev_loss = evaluate(dev_batch, loss_fn, args, "dev")
                test_precision, test_recall, test_f1, test_loss = evaluate(test_batch, loss_fn, args, "test")

                if epoch == 0 or dev_f1 > best_result:
                    best_result = dev_f1
                    test_precision_best = test_precision
                    test_recall_best = test_recall
                    test_f1_best = test_f1
                    #torch.save(model.state_dict(), output_dir + "/" + model_file)
                    logger.info("Best result on dev saved!!!")


                saved_file.save("{} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}".format(epoch,
                                                                                                                                      train_loss / steps,
                                                                                                                                      dev_loss,
                                                                                                                                      test_loss,
                                                                                                                                      dev_precision,
                                                                                                                                      dev_recall,
                                                                                                                                      dev_f1,
                                                                                                                                      test_precision,
                                                                                                                                      test_recall,
                                                                                                                                      test_f1))


        saved_file.save("best test results: precision: {:.4f} \t recall: {:.4f} \t f1: {:.4f}".format(test_precision_best,
                                                                                                      test_recall_best,
                                                                                                      test_f1_best))