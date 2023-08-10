import torch
import torch.distributed as dist
from torchmetrics import Metric

from ptunifier.metrics.pycocoevalcap.eval import compute_scores
from ptunifier.metrics.jb_scorers.scores import compute_jb_scores


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        preds = logits.argmax(dim=-1)
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total


class VQARADScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("close_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("close_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("open_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("open_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.best_score = 0
        self.best_close_score = 0
        self.best_open_score = 0

    def update(self, logits, target, types=None):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        close_scores = scores[types == 0]
        open_scores = scores[types == 1]

        self.close_score += close_scores.sum()
        self.close_total += len(close_scores)
        self.open_score += open_scores.sum()
        self.open_total += len(open_scores)

        self.score += scores.sum()
        self.total += len(scores)

    def compute(self):
        score = self.score / self.total
        return score

    def get_best_score(self):
        self.sync()
        score = self.score / self.total
        if score > self.best_score:
            self.best_score = score
            self.best_close_score = self.close_score / self.close_total if self.close_total != 0 else 0
            self.best_open_score = self.open_score / self.open_total if self.open_total != 0 else 0
        self.unsync()
        return self.best_score

    def get_best_close_score(self):
        return self.best_close_score

    def get_best_open_score(self):
        return self.best_open_score


def dist_reduce_objects_dx(result):
    output = [None] * dist.get_world_size()
    dist.all_gather_object(output, result)
    flattened_output = [text for text_list in output for text in text_list]
    return flattened_output


class COCOCaptionScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("res", default=[], dist_reduce_fx=dist_reduce_objects_dx)
        self.add_state("gts", default=[], dist_reduce_fx=dist_reduce_objects_dx)

    def update(self, re, gt):
        self.res.extend(re)
        self.gts.extend(gt)

    def compute(self):
        if dist.get_rank() == 0:
            scores = [compute_scores(gts={str(i): [gt] for i, gt in enumerate(self.gts)},
                                     res={str(i): [re] for i, re in enumerate(self.res)})]
        else:
            scores = [None]
        dist.broadcast_object_list(scores, src=0)
        return scores[0]


class JBScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("res", default=[], dist_reduce_fx=dist_reduce_objects_dx)
        self.add_state("gts", default=[], dist_reduce_fx=dist_reduce_objects_dx)
        # Reference Time (s)
        # {'BLEU': 0.29,
        #  'CIDERD': 0.49,
        #  'MAUVE': 202.75,
        #  'METEOR': 6.91,
        #  'ROUGE1': 0.31,
        #  'ROUGE2': 0.31,
        #  'ROUGEL': 0.33,
        #  'bertscore': 10.65,
        #  'chexbert': 26.39,
        #  'radentitymatchexact': 5.54,
        #  'radentitynli': 57.65,
        #  'radgraph': 106.06}
        self.all_metrics = ["BLEU", "METEOR", "CIDERD", "ROUGE1", "ROUGE2", "ROUGEL",
                            "bertscore", "chexbert",  # "MAUVE",
                            "radentitymatchexact", "radentitynli", "radgraph"]

    def update(self, re, gt):
        self.res.extend(re)
        self.gts.extend(gt)

    def compute(self):
        if dist.get_rank() == 0:
            scores = [compute_jb_scores(metrics=self.all_metrics, refs=self.gts, hyps=self.res)]
        else:
            scores = [None]
        dist.broadcast_object_list(scores, src=0)
        return scores[0]
