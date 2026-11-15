"""python inference.py --model qat_int8 --image photo.png
Omit --image to run the first CIFAR-10 test image with its true label.
"""
import argparse
from PIL import Image
from common import *
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',default='fp32',choices=['fp32','ptq_int8','qat_int8','unstructured','structured']); p.add_argument('--image'); a=p.parse_args(); setup()
    if a.image:
        im=np.asarray(Image.open(a.image).convert('RGB').resize((32,32)),dtype=np.float32)/255
        x=torch.from_numpy(im.transpose(2,0,1)).unsqueeze(0)
        x=(x-torch.tensor([.4914,.4822,.4465]).view(1,3,1,1))/torch.tensor([.2470,.2435,.2616]).view(1,3,1,1)
    else:
        *_,xt,yt=data(); x=xt[:1]; print('True:',CLASSES[int(yt[0])])
    m=torch.jit.load(str(MODELS/(a.model+'.pt'))).eval()
    with torch.inference_mode(): prob=m(x).softmax(1)[0]; k=int(prob.argmax())
    print(dict(model=a.model,prediction=CLASSES[k],confidence=float(prob[k])))
