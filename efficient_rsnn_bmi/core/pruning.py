import torch

def prune_model(model, prune_percentage):
    mask = {}
    for name, param in model.named_parameters():
        # print(name, ":", param)
        if 'weight' in name:
            _, indices = torch.topk(torch.abs(param.data.flatten()), int(param.data.numel() * (prune_percentage)),
                                    largest=False)
            mask[name] = torch.ones_like(param.data).flatten()
            mask[name][indices] = 0
            mask[name] = mask[name].view_as(param.data)
            param.data.mul_(mask[name])
    return mask