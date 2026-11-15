import argparse
from common import *
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=3); a=p.parse_args()
    setup(); x,y,xv,yv,_,_=data(); m=load_float().train(); m.fuse(qat=True)
    m.qconfig=tq.get_default_qat_qconfig('onednn'); tq.prepare_qat(m,inplace=True)
    print('Fake quantizers:',sum(isinstance(z,tq.FakeQuantize) for z in m.modules()),flush=True)
    train(m,x,y,xv,yv,a.epochs,.002,'qat'); tq.convert(m.eval(),inplace=True)
    torch.jit.script(m).save(str(MODELS/'qat_int8.pt'))
    print('QAT INT8 saved',flush=True)
