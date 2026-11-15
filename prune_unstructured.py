from common import *
from torch.nn.utils import prune
if __name__=='__main__':
    setup(); x,y,xv,yv,_,_=data(); m=load_float()
    modules=[(z,'weight') for z in m.modules() if isinstance(z,(nn.Conv2d,nn.Linear))]
    history=[]
    for iteration in range(3):
        prune.global_unstructured(modules,pruning_method=prune.L1Unstructured,amount=.2)
        zero=sum(int((z.weight==0).sum()) for z,_ in modules); total=sum(z.weight.numel() for z,_ in modules)
        acc,_,_=evaluate(m,xv,yv)
        history.append(dict(iteration=iteration+1,weight_zeros=zero,weight_count=total,sparsity=zero/total,validation_accuracy_before_recovery=acc))
        print(history[-1],flush=True)
        train(m,x,y,xv,yv,1,.005,'unstructured_round'+str(iteration+1))
    masks={name:z.weight_mask.detach().clone() for name,z in m.named_modules() if hasattr(z,'weight_mask')}
    for z,_ in modules: prune.remove(z,'weight')
    assert not any('mask' in k or 'orig' in k for k in m.state_dict())
    save_float(m,'unstructured'); dump_json(RESULTS/'pruning_iterations.json',history)
    print('Permanent zero weights:',sum(int((z.weight==0).sum()) for z,_ in modules),flush=True)
