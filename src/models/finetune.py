"""Member 3, W3.5 — attach ClassificationHead to a (pretrained or
randomly-initialized) encoder and fine-tune on the labeled set.

CRITICAL: the random-init (M1) and pretrained (M2) runs must be
IDENTICAL in every respect except where the weights started -- same
architecture, fine-tuning recipe, data, schedule, seeds. Any other
difference invalidates Result 1 (see team doc W3.5)."""


def finetune(encoder, head, train_loader, val_loader, max_epochs: int):
    raise NotImplementedError
