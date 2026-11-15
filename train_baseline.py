import argparse
from common import *
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=15); a=p.parse_args()
    setup(); x,y,xv,yv,_,_=data(); m=ResNet18()
    print('Training parameters:',sum(p.numel() for p in m.parameters()),flush=True)
    train(m,x,y,xv,yv,a.epochs,.08,'baseline'); save_float(m,'fp32')
