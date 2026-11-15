"""Artifact checks: reloading, residual dimensions, real sparsity, CSV consistency."""
import hashlib
import onnx
from common import *
from benchmark import NAMES
if __name__=='__main__':
    setup(); rows=json.loads((RESULTS/'benchmark.json').read_text())
    predictions=list(csv.DictReader((RESULTS/'predictions.csv').open()))
    checks=[]
    for name in NAMES:
        m=torch.jit.load(str(MODELS/(name+'.pt'))).eval()
        with torch.inference_mode(): assert m(torch.randn(2,3,32,32)).shape==(2,10)
        selected=[p for p in predictions if p['model']==name]
        assert len(selected)==10000
        r=next(r for r in rows if r['model']==name)
        assert sum(int(p['correct']) for p in selected)==r['correct']
        onnx.checker.check_model(onnx.load(str(MODELS/(name+'.onnx'))))
        checks.append(name+': checkpoint reload, output shape, prediction counts and ONNX check passed')
    base=load_float(); u=load_float('unstructured'); s=load_float('structured')
    assert not any('mask' in k or 'orig' in k for k in u.state_dict())
    assert sum(p.numel() for p in s.parameters())<sum(p.numel() for p in base.parameters())
    for b in s.blocks:
        assert b.conv1.out_channels==b.bn1.num_features==b.conv2.in_channels
    sparse=next(r for r in rows if r['model']=='unstructured')
    assert .487<sparse['conv_linear_weight_sparsity']<.489
    for name in ['ptq_int8','qat_int8']:
        text=str(torch.jit.load(str(MODELS/(name+'.pt'))).inlined_graph)
        assert 'quantized::conv2d' in text
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in MODELS.iterdir() if p.is_file()}
    dump_json(RESULTS/'verification.json',dict(status='PASS',checks=checks+['48.8% permanent unstructured weight sparsity','Structured residual dimensions preserved; parameters physically reduced','PTQ and QAT contain integer convolution operators'],model_sha256=hashes))
    print('PASS:',len(checks)+3,'checks;',len(hashes),'model artifacts',flush=True)
