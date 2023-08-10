import os
import time
from pprint import pprint
from .tokenizer.ptbtokenizer import PTBTokenizer
from .NLG.rouge.rouge import Rouge
from .NLG.bleu.bleu import Bleu
from .NLG.meteor.meteor import Meteor
from .NLG.ciderD.ciderD import CiderD
from .NLG.bertscore.bertscore import BertScore
from .NLG.mauve_.mauve_ import MauveScorer
from .CheXbert.chexbert import CheXbert
from .RadEntityMatchExact.RadEntityMatchExact import RadEntityMatchExact
from .RadEntityNLI.RadEntityNLI import RadEntityNLI
from .RadGraph.RadGraph import RadGraph

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def compute_jb_scores(metrics, refs, hyps, device="cuda"):
    # print('tokenization...')
    tokenizer = PTBTokenizer()
    refs = tokenizer.tokenize_list(refs)
    hyps = tokenizer.tokenize_list(hyps)

    scores = dict()
    times = dict()

    for metric in metrics:
        t = time.time()
        # Iterating over metrics
        if metric == 'BLEU':
            scores["BLEU1"] = Bleu(n=1)(refs, hyps)[0]
            scores["BLEU2"] = Bleu(n=2)(refs, hyps)[0]
            scores["BLEU3"] = Bleu(n=3)(refs, hyps)[0]
            scores["BLEU4"] = Bleu(n=4)(refs, hyps)[0]
            times[metric] = round(time.time() - t, 2)

        elif metric == 'METEOR':
            scores["METEOR"] = Meteor()(refs, hyps)[0]
            times[metric] = round(time.time() - t, 2)

        elif metric == 'CIDERD':
            scores["CIDERD"] = CiderD()(refs, hyps)[0]
            times[metric] = round(time.time() - t, 2)

        elif metric in ['ROUGE1', 'ROUGE2', 'ROUGEL']:
            scores[metric] = Rouge(rouges=[metric.lower()])(refs, hyps)[0]
            times[metric] = round(time.time() - t, 2)

        elif metric == 'bertscore':
            scores["bertscore"] = BertScore(device=device)(refs, hyps)[0]
            times[metric] = round(time.time() - t, 2)

        elif metric == 'MAUVE':
            scores["MAUVE"] = round(MauveScorer(device=device)(refs, hyps) * 100, 2)
            times[metric] = round(time.time() - t, 2)

        elif metric == 'chexbert':
            accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = CheXbert(device=device)(hyps, refs)
            scores["chexbert-5_micro avg_f1-score"] = chexbert_5["micro avg"]["f1-score"]
            scores["chexbert-all_micro avg_f1-score"] = chexbert_all["micro avg"]["f1-score"]
            times[metric] = round(time.time() - t, 2)

        elif metric == 'radentitymatchexact':
            scores["radentitymatchexact"] = RadEntityMatchExact()(refs, hyps)[0]
            times[metric] = round(time.time() - t, 2)

        elif metric == 'radentitynli':
            scores["radentitynli"] = RadEntityNLI(device=device)(refs, hyps)[0]
            times[metric] = round(time.time() - t, 2)

        elif metric == 'radgraph':
            scores["radgraph_simple"], scores["radgraph_partial"], scores["radgraph_complete"] = \
                RadGraph(reward_level="full", device=device)(refs=refs, hyps=hyps)[0]
            times[metric] = round(time.time() - t, 2)

        else:
            print("Metric not implemented: {}".format(metric))

    # print("Time for evaluation metrics:")
    # pprint(times)

    return scores


if __name__ == '__main__':
    hyps = [h.strip() for h in open("test_best-1_835456_hyps.txt").readlines()]
    refs = [r.strip() for r in open("test_best-1_835456_refs.txt").readlines()]

    compute_jb_scores(metrics=["BLEU", "METEOR", "CIDERD", "ROUGE1", "ROUGE2", "ROUGEL",
                               "bertscore", "MAUVE", "chexbert",
                               "radentitymatchexact", "radentitynli", "radgraph"],
                      refs=refs,
                      hyps=hyps)
