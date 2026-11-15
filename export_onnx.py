"""Export executable PyTorch models to ONNX and validate with ONNX Runtime CPU."""
import onnx, onnxruntime as ort
from common import *
from benchmark import NAMES
if __name__=='__main__':
    setup(); *_,x,y=data(); records=[]
    for name in NAMES:
        model=torch.jit.load(str(MODELS/(name+'.pt'))).eval()
        output=MODELS/(name+'.onnx')
        torch.onnx.export(model,x[:1],str(output),opset_version=17,input_names=['input'],output_names=['logits'],dynamic_axes={'input':{0:'batch'},'logits':{0:'batch'}},dynamo=False)
        graph=onnx.load(str(output)); onnx.checker.check_model(graph)
        options=ort.SessionOptions(); options.intra_op_num_threads=torch.get_num_threads(); options.inter_op_num_threads=1
        session=ort.InferenceSession(str(output),sess_options=options,providers=['CPUExecutionProvider']); pred=[]; max_error=0.
        with torch.inference_mode():
            for i in range(0,len(y),256):
                xb=x[i:i+256]; out=session.run(None,{'input':xb.numpy()})[0]; pred.extend(out.argmax(1).tolist())
                max_error=max(max_error,float(np.max(np.abs(out-model(xb).numpy()))))
        row=dict(model=name,onnx_bytes=output.stat().st_size,test_count=len(y),accuracy_pct=float(np.mean(np.array(pred)==y.numpy())*100),max_absolute_logit_difference=max_error,quantize_linear_nodes=sum(n.op_type=='QuantizeLinear' for n in graph.graph.node),dequantize_linear_nodes=sum(n.op_type=='DequantizeLinear' for n in graph.graph.node))
        records.append(row); print(row,flush=True)
    dump_json(RESULTS/'onnx_validation.json',records)
    with (RESULTS/'onnx_validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
