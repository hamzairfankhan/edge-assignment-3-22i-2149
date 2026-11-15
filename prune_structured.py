from common import *
def reconstruct(original):
    # Keep strongest 50% of each block's internal channels by conv1 L1 norm.
    # Conv2 input columns and BN1 entries use exactly the same indices.
    widths=[b.conv1.out_channels//2 for b in original.blocks]
    compact=ResNet18(widths); state=compact.state_dict(); old=original.state_dict(); selected={}
    for key in state:
        if state[key].shape==old[key].shape: state[key]=old[key].clone()
    for i,(b,c) in enumerate(zip(original.blocks,compact.blocks)):
        idx=b.conv1.weight.detach().abs().sum((1,2,3)).topk(widths[i]).indices.sort().values
        prefix=f'blocks.{i}.'; selected[str(i)]=idx.tolist()
        state[prefix+'conv1.weight']=b.conv1.weight.detach()[idx].clone()
        state[prefix+'conv2.weight']=b.conv2.weight.detach()[:,idx].clone()
        for suffix in ['weight','bias','running_mean','running_var']:
            state[prefix+'bn1.'+suffix]=getattr(b.bn1,suffix).detach()[idx].clone()
    compact.load_state_dict(state); return compact,widths,selected
if __name__=='__main__':
    setup(); x,y,xv,yv,_,_=data(); original=load_float(); m,widths,indices=reconstruct(original)
    before=evaluate(m,xv,yv)[0]; print('Structured before recovery validation:',before,flush=True)
    train(m,x,y,xv,yv,3,.01,'structured'); save_float(m,'structured',widths)
    dump_json(RESULTS/'channel_selection.json',dict(hidden_widths=widths,retained_indices=indices,validation_before_recovery=before,validation_after_recovery=evaluate(m,xv,yv)[0]))
    print('Physically reconstructed parameters:',sum(p.numel() for p in m.parameters()),flush=True)
