import argparse
import onnxruntime as ort
from common import *
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',default='structured'); a=p.parse_args(); setup(); *_,x,y=data()
    options=ort.SessionOptions(); options.intra_op_num_threads=4
    session=ort.InferenceSession(str(MODELS/(a.model+'.onnx')),sess_options=options,providers=['CPUExecutionProvider'])
    out=session.run(None,{'input':x[:1].numpy()})[0]
    print(dict(model=a.model,predicted_class=int(out.argmax()),true_class=int(y[0]),logits=out.tolist()))
