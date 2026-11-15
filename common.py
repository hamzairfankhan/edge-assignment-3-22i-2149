"""CIFAR-10 ResNet-18 (16-channel stem) and shared reproducible utilities."""
from pathlib import Path
import csv, json, os, random, time
import numpy as np
import torch
from torch import nn
from torch.ao import quantization as tq
from torchvision.datasets import CIFAR10

ROOT=Path(__file__).resolve().parent
MODELS=ROOT/'models'; RESULTS=ROOT/'results'
SEED=2149
def setup():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(int(os.environ.get('EDGE_THREADS','4')))
    torch.backends.quantized.engine='onednn'
    MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)

class Block(nn.Module):
    def __init__(self,cin,cout,stride=1,hidden=0):
        super().__init__(); hidden=hidden or cout
        self.conv1=nn.Conv2d(cin,hidden,3,stride,1,bias=False)
        self.bn1=nn.BatchNorm2d(hidden); self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(hidden,cout,3,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(cout)
        self.shortcut=nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout)) if cin!=cout or stride!=1 else nn.Identity()
        self.add=nn.quantized.FloatFunctional()
    def forward(self,x):
        y=self.relu(self.bn1(self.conv1(x))); y=self.bn2(self.conv2(y))
        return self.add.add_relu(y,self.shortcut(x))
    def fuse(self,qat=False):
        f=tq.fuse_modules_qat if qat else tq.fuse_modules
        f(self,[['conv1','bn1','relu'],['conv2','bn2']],inplace=True)
        if isinstance(self.shortcut,nn.Sequential): f(self.shortcut,[['0','1']],inplace=True)

class ResNet18(nn.Module):
    """18 weighted layers: stem + 8 two-convolution residual blocks + FC.

    Uses a CIFAR 3x3 stride-1 stem and stage widths 16/32/64/128.
    Internal hidden widths may shrink without changing residual addition shapes.
    """
    def __init__(self,hidden=None):
        super().__init__(); hidden=hidden or [16,16,32,32,64,64,128,128]
        self.quant=tq.QuantStub(); self.conv=nn.Conv2d(3,16,3,1,1,bias=False)
        self.bn=nn.BatchNorm2d(16); self.relu=nn.ReLU(); blocks=[]; cin=16
        for i,cout in enumerate([16,16,32,32,64,64,128,128]):
            blocks.append(Block(cin,cout,2 if i in [2,4,6] else 1,hidden[i])); cin=cout
        self.blocks=nn.Sequential(*blocks); self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(128,10); self.dequant=tq.DeQuantStub()
    def forward(self,x):
        x=self.quant(x); x=self.relu(self.bn(self.conv(x))); x=self.blocks(x)
        x=self.pool(x).flatten(1); return self.dequant(self.fc(x))
    def fuse(self,qat=False):
        (tq.fuse_modules_qat if qat else tq.fuse_modules)(self,[['conv','bn','relu']],inplace=True)
        for b in self.blocks: b.fuse(qat)

def data():
    path=os.environ.get('CIFAR_ROOT',str(ROOT/'data'))
    tr=CIFAR10(path,train=True,download=True); te=CIFAR10(path,train=False,download=True)
    mean=torch.tensor([.4914,.4822,.4465]).view(1,3,1,1)
    std=torch.tensor([.2470,.2435,.2616]).view(1,3,1,1)
    def tensor(d): return (torch.tensor(d.data.transpose(0,3,1,2),dtype=torch.float32)/255-mean)/std,torch.tensor(d.targets)
    x,y=tensor(tr); xt,yt=tensor(te)
    order=torch.randperm(50000,generator=torch.Generator().manual_seed(SEED))
    # Fixed holdout: 45,000 training, 5,000 validation; calibration only training.
    return x[order[:45000]],y[order[:45000]],x[order[45000:]],y[order[45000:]],xt,yt

@torch.inference_mode()
def evaluate(model,x,y,batch=256):
    model.eval(); predictions=[]; losses=[]
    for i in range(0,len(y),batch):
        z=model(x[i:i+batch]); predictions.append(z.argmax(1)); losses.append(nn.functional.cross_entropy(z,y[i:i+batch],reduction='sum').item())
    p=torch.cat(predictions)
    return float((p==y).float().mean()*100),sum(losses)/len(y),p

def train(model,x,y,xv,yv,epochs,lr,label,mask=None):
    opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    hist=[]
    for epoch in range(epochs):
        start=time.perf_counter(); model.train(); total=0.; correct=0
        for idx in torch.randperm(len(y)).split(256):
            xb=x[idx].clone(); flip=torch.rand(len(idx))<.5; xb[flip]=xb[flip].flip(3)
            opt.zero_grad(); z=model(xb); loss=nn.functional.cross_entropy(z,y[idx]); loss.backward(); opt.step()
            if mask:
                with torch.no_grad():
                    for name,m in model.named_modules():
                        if name in mask: m.weight.mul_(mask[name])
            total+=loss.item()*len(idx); correct+=(z.argmax(1)==y[idx]).sum().item()
        sched.step(); acc,vl,_=evaluate(model,xv,yv)
        row=dict(stage=label,epoch=epoch+1,train_loss=total/len(y),train_accuracy=100*correct/len(y),validation_accuracy=acc,validation_loss=vl,seconds=time.perf_counter()-start)
        hist.append(row); print(json.dumps(row),flush=True)
        with (RESULTS/(label+'_history.json')).open('w') as f: json.dump(hist,f,indent=2)
    return model.eval()

def save_float(model,name,hidden=None):
    torch.save({'state_dict':model.state_dict(),'hidden':hidden,'architecture':'CIFAR ResNet-18 width16','seed':SEED},MODELS/(name+'.pth'))
    torch.jit.script(model.eval()).save(str(MODELS/(name+'.pt')))

def load_float(name='fp32'):
    ck=torch.load(MODELS/(name+'.pth'),map_location='cpu',weights_only=True)
    m=ResNet18(ck['hidden']); m.load_state_dict(ck['state_dict']); return m.eval()

def dump_json(path,obj):
    path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
