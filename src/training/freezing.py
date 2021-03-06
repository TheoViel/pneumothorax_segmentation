def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def unfreeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()
        m.weight.requires_grad = True
        m.bias.requires_grad = True


def freeze(m):
    for param in m.parameters():
        param.requires_grad = False
    unfreeze_bn(m)


def unfreeze(m):
    for param in m.parameters():
        param.requires_grad = True


def freeze_encoder(model):
    for n, p in list(model.named_parameters()):
        if "encoder" in n:
            p.requires_grad = False
    model.apply(unfreeze_bn)


def unfreeze_encoder(model):
    for n, p in list(model.named_parameters()):
        if "encoder" in n:
            p.requires_grad = True

