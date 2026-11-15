from common import *
from calibrate import calibrate
if __name__=='__main__':
    setup(); x,*_=data(); m=load_float(); m.fuse(); m.qconfig=tq.get_default_qconfig('onednn')
    tq.prepare(m,inplace=True); calibrate(m,x); tq.convert(m,inplace=True)
    torch.jit.script(m).save(str(MODELS/'ptq_int8.pt'))
    print('PTQ saved; integer convolution:',type(m.blocks[0].conv1),flush=True)
