from util import *
from params import *
from metric import *
from imports import *
from training.radam import *
from training.losses import *
from training.freezing import *
from training.learning_rate import *


def update_average(model1, model2, decay=0.99):
    par1 = model1.state_dict()
    par2 = model2.state_dict()

    with torch.no_grad():
        for k in par1.keys():
            par1[k].data.copy_(par1[k].data * decay + par2[k].data * (1 - decay))
    

def fit_seg(model, model_shadow, train_dataset, val_dataset, epochs=50, batch_size=32, use_aux_clf=False, acc_steps=1,
            warmup_prop=0.1, lr=1e-3, schedule='cosine', min_lr=1e-5, smooth_masks=False, use_ema=False, ema_decay=0.99,
            verbose=1, verbose_eval=10, cp=False, model_name='model'):
       
    best_dice = 0
    avg_val_loss = 1000
    lr_init = lr


    params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    encoder_params = [(n, p) for n, p in params if any(nd in n for nd in ['encoder', 'logit', 'center'])]
    opt_params = [
        {'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-4},
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay) and 'decoder' in n],
         'weight_decay': 1e-2},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = RAdam(opt_params, lr=lr)
#     optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    if schedule == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs - ceil(epochs * warmup_prop), eta_min=min_lr)
    elif schedule == 'reduce_lr':
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=ceil(5 / verbose_eval) - 1)

    loss_seg = lov_loss #BCEWithLogitsLoss(reduction='mean') #lov_loss  # hck_focal_loss
    loss_clf = BCEWithLogitsLoss(reduction='mean')
    loss_clf_w = 0.1

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               drop_last=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=VAL_BS, shuffle=False, num_workers=NUM_WORKERS)
    
    filter_size = train_dataset[0][0].shape[-1] // 40 * 2 + 1
    filters = (torch.ones((4, 1, filter_size, filter_size)) / (filter_size ** 2)).cuda().detach()
        
    for epoch in range(epochs):
        model.train()
        if use_ema:
            model_shadow.train()
            
        if batch_size < 4:
            model.apply(freeze_bn)

        avg_loss = 0
        start_time = time.time()

        lr = schedule_lr(optimizer, epoch, scheduler, scheduler_name=schedule, avg_val_loss=avg_val_loss,
                         epochs=epochs, warmup_prop=warmup_prop, lr_init=lr_init, min_lr=min_lr,
                         verbose_eval=verbose_eval)
        
        optimizer.zero_grad()
        for step, (x, y_batch, fault_batch) in enumerate(train_loader):
        
            if smooth_masks:
                with torch.no_grad():
                    y_batch = F.conv2d(y_batch.cuda(), filters, groups=4, padding=(filter_size - 1)//2)

            y_pred, fault_pred = model(x.cuda())
            
            if use_aux_clf:
                loss = loss_seg(y_pred, y_batch.cuda()) + loss_clf(fault_pred,
                                                                   fault_batch.cuda().float()) * loss_clf_w
            else:
                loss = loss_seg(y_pred, y_batch.cuda())
            loss.backward()
            avg_loss += loss.item() / len(train_loader)
            
            if step % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                if use_ema:
                    update_average(model_shadow, model, decay=ema_decay)

        del y_pred, fault_pred
        torch.cuda.empty_cache()

        model.eval()
        if use_ema:
            model_shadow.eval()
            
        avg_val_loss = 0.
        val_dice = 0.

        if (epoch + 1) % verbose_eval == 0 or (epoch + 1) == epochs:
            with torch.no_grad():
                for x, y_batch, fault_batch in val_loader:
                    bs, c, h, w, = y_batch.size()
                    
                    if use_ema:
                        y_pred, fault_pred = model_shadow(x.cuda())
                    else:
                        y_pred, fault_pred = model(x.cuda())

                    if use_aux_clf:
                        loss = loss_seg(y_pred.detach(), y_batch.cuda()) + loss_clf(fault_pred.detach(),
                                                                                    fault_batch.cuda().float()) * loss_clf_w
                    else:
                        loss = loss_seg(y_pred.detach(), y_batch.cuda())
                    avg_val_loss += loss.item() / len(val_loader)

                    probs = torch.sigmoid(y_pred.detach())
                    val_dice += dice_th(probs.contiguous().cpu(), y_batch, resize=True) / len(val_loader)

                del probs, y_batch, y_pred, fault_pred
                torch.cuda.empty_cache()
             
            if val_dice > best_dice and cp:
                save_model_weights(model, f"{model_name}_cp.pt", verbose=0)
                if use_ema:
                    save_model_weights(model_shadow, f"{model_name}_shadow_cp.pt", verbose=0)
                best_dice = val_dice
           
        elapsed_time = time.time() - start_time

        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            print(f'Epoch {epoch + 1}/{epochs}     lr={lr:.1e}     t={elapsed_time:.0f}s     loss={avg_loss:.4f}     ',
                  end='')
            if verbose_eval and ((epoch + 1) % verbose_eval == 0 or (epoch + 1) == epochs):
                print(f'dice={val_dice:.5f}     val_loss={avg_val_loss:.4f}     ', end='\n')

            else:
                print(' ', end='\n')

        if lr < min_lr:
            print(f'Reached low enough learning rate ({min_lr:.1e}), interrupting...')
            break
