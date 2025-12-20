from typing import Optional

import torch
import torch.nn.functional as F


def _extract(a: torch.Tensor, t: torch.LongTensor, x_shape: torch.Size) -> torch.Tensor:
    out = a.index_select(0, t)
    while out.dim() < len(x_shape):
        out = out.view(-1, *([1] * (len(x_shape) - 1)))
    return out


def noise_estimation_loss(
    model,
    x0: torch.Tensor,
    t: torch.LongTensor,
    e: torch.Tensor,
    b: torch.Tensor,
    keepdim: bool = False,
    **kwargs,
):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def discrete_token_loss(
    model,
    x0: torch.Tensor,
    t: torch.LongTensor,
    e: torch.Tensor,
    b: torch.Tensor,
    keepdim: bool = False,
    *,
    mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    if x0.dim() == 3:
        x0_ids = x0.argmax(dim=-1)
    else:
        x0_ids = x0.long()

    model_module = getattr(model, "module", model)
    vocab_size = int(getattr(model_module, "vocab_size", int(x0_ids.max().item()) + 1))

    alpha_bar = (1 - b).cumprod(dim=0)
    keep_prob = _extract(alpha_bar, t, x0_ids.shape)
    noise = torch.randint(0, vocab_size, x0_ids.shape, device=x0_ids.device)
    keep_mask = torch.rand_like(x0_ids.float()) < keep_prob
    x_t = torch.where(keep_mask, x0_ids, noise)

    pred_x0, score = model(x_t, t.float(), mask=mask)

    pred_loss = F.cross_entropy(
        pred_x0.view(-1, pred_x0.size(-1)),
        x0_ids.view(-1),
        reduction="none",
    ).view_as(x0_ids)
    if mask is not None:
        pred_loss = pred_loss * mask

    if keepdim:
        loss = pred_loss.sum(dim=tuple(range(1, pred_loss.dim())))
    else:
        denom = mask.sum() if mask is not None else pred_loss.numel()
        loss = pred_loss.sum() / denom

    if score is not None:
        score_loss = F.cross_entropy(
            score.view(-1, score.size(-1)),
            x_t.view(-1),
            reduction="none",
        ).view_as(x0_ids)
        if mask is not None:
            score_loss = score_loss * mask
        if keepdim:
            loss = loss + score_loss.sum(dim=tuple(range(1, score_loss.dim())))
        else:
            denom = mask.sum() if mask is not None else score_loss.numel()
            loss = loss + score_loss.sum() / denom

    return loss


loss_registry = {
    'simple': noise_estimation_loss,
    'discrete': discrete_token_loss,
}
