"""Test all five executable checkpoints; batch-1 CPU latency is measured separately."""
import platform, sys
from common import *
NAMES=['fp32','ptq_int8','qat_int8','unstructured','structured']
@torch.inference_mode()
def latency(model,sample):
    for _ in range(30): model(sample)
    times=[]
    for _ in range(200):
        t=time.perf_counter_ns(); model(sample); times.append((time.perf_counter_ns()-t)/1e6)
    return times
def complexity(model):
    macs=[0]; handles=[]
    def hook(m,args,out):
        if isinstance(m,nn.Conv2d): macs[0]+=out.numel()*m.in_channels*m.kernel_size[0]*m.kernel_size[1]//m.groups
        elif isinstance(m,nn.Linear): macs[0]+=out.numel()*m.in_features
    for mod in model.modules():
        if isinstance(mod,(nn.Conv2d,nn.Linear)): handles.append(mod.register_forward_hook(hook))
    with torch.inference_mode(): model(torch.zeros(1,3,32,32))
    for h in handles: h.remove()
    return macs[0]
if __name__=='__main__':
    setup(); _,_,_,_,x,y=data(); rows=[]; predictions=[]; latency_rows=[]
    for name in NAMES:
        path=MODELS/(name+'.pt'); model=torch.jit.load(str(path)).eval()
        start=time.perf_counter(); accuracy,loss,p=evaluate(model,x,y); test_seconds=time.perf_counter()-start
        samples=latency(model,x[:1]); reference=load_float('structured' if name=='structured' else 'fp32')
        params=sum(z.numel() for z in reference.parameters()); weights=[z.weight for z in reference.modules() if isinstance(z,(nn.Conv2d,nn.Linear))]
        if name=='unstructured':
            weights=[z.weight for z in load_float(name).modules() if isinstance(z,(nn.Conv2d,nn.Linear))]
        row=dict(model=name,test_count=len(y),correct=int((p==y).sum()),accuracy_pct=accuracy,test_loss=loss,checkpoint_bytes=path.stat().st_size,parameters=params,conv_linear_weight_sparsity=sum(int((w==0).sum()) for w in weights)/sum(w.numel() for w in weights),dense_macs=complexity(reference),batch1_median_ms=float(np.median(samples)),batch1_p95_ms=float(np.percentile(samples,95)),batch256_test_seconds=test_seconds)
        if name in ['ptq_int8','qat_int8']:
            # Sparsity describes float pruning, not rounding-induced INT8 zeros.
            row['conv_linear_weight_sparsity']=None
        rows.append(row); print(json.dumps(row),flush=True)
        predictions.extend(dict(model=name,test_index=i,true_label=int(y[i]),prediction=int(p[i]),correct=int(p[i]==y[i]),model_batch1_median_ms=row['batch1_median_ms'],model_batch1_p95_ms=row['batch1_p95_ms'],checkpoint_bytes=row['checkpoint_bytes']) for i in range(len(y)))
        latency_rows.extend(dict(model=name,repeat=i+1,latency_ms=v) for i,v in enumerate(samples))
    for filename,records in [('benchmark.csv',rows),('predictions.csv',predictions),('latency_samples.csv',latency_rows)]:
        with (RESULTS/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    dump_json(RESULTS/'benchmark.json',rows)
    dump_json(RESULTS/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),processor=platform.processor(),threads=torch.get_num_threads(),quantization_backend=torch.backends.quantized.engine,seed=SEED,train_count=45000,validation_count=5000,test_count=10000,calibration_count=1024,baseline_epochs=15,qat_epochs=3,pruning_iterations=3,structured_recovery_epochs=3,warmup=30,latency_repeats=200))
