"""Temporarily relocate an idle decoder layer during native visual attention."""
import json
from pathlib import Path
import time


def stage_first_decoder_during_vision(model, torch, trace_file):
    """Preserve CUDA fp16 computation, weights and native forward functions.

    The Qwen visual encoder finishes before the first decoder layer executes.
    Stage that unused layer on CUDA1 while vision runs on CUDA0, then restore
    it before returning to native generation. This requires serial inference.
    The caller must verify exact historical requests before formal evaluation.
    """
    if (model.training or model.config.model_type != 'qwen2_5_vl'
        or len(model.model.layers) != 28 or getattr(model.visual, '_causalact_staged', False)):
        raise RuntimeError('Unexpected or already staged UI-TARS architecture')
    if not set(str(v) for v in model.hf_device_map.values()).issubset({'0','1','cuda:0','cuda:1'}):
        raise RuntimeError('Staging requires the verified two-CUDA placement')
    layer=model.model.layers[0]
    parameters=list(layer.parameters())
    if not parameters or any(p.device.type!='cuda' or p.device.index!=0 or p.dtype!=torch.float16 for p in parameters):
        raise RuntimeError('First decoder layer must start on CUDA0 in fp16')
    weight_bytes=sum(p.numel()*p.element_size() for p in parameters)
    original=model.visual.forward
    verified_once=False
    busy=False

    def forward(*args, **kwargs):
        nonlocal verified_once,busy
        if busy:
            raise RuntimeError('Concurrent native vision requests are unsupported')
        busy=True
        before=None
        record=dict(started_unix=time.time(),weight_bytes=weight_bytes,
            staged_from='cuda:0',staged_to='cuda:1',CPU_model_computation=False)
        try:
            for index in [0,1]:
                with torch.cuda.device(index):torch.cuda.empty_cache()
            if torch.cuda.mem_get_info(1)[0] < weight_bytes+256*1024**2:
                raise RuntimeError('No destination memory for idle decoder weights')
            if not verified_once:
                # Verification copies are host-side only, never model inputs.
                before={name:value.detach().cpu().clone() for name,value in layer.state_dict().items()}
            layer.to('cuda:1')
            torch.cuda.synchronize(1)
            with torch.cuda.device(0):torch.cuda.empty_cache()
            return original(*args,**kwargs)
        finally:
            try:
                layer.to('cuda:0')
                torch.cuda.synchronize(0)
                if any(p.device.type!='cuda' or p.device.index!=0 or p.dtype!=torch.float16 for p in layer.parameters()):
                    raise RuntimeError('Decoder weights were not restored before native decode')
                if before is not None:
                    after=layer.state_dict()
                    if set(after)!=set(before) or not all(torch.equal(before[k],v.detach().cpu()) for k,v in after.items()):
                        raise RuntimeError('Decoder weight values changed during CUDA transfer')
                    verified_once=True
                    record['first_roundtrip_all_state_tensors_exact']=True
                record.update(restored_to='cuda:0',native_forward_unchanged=True,finished_unix=time.time())
                with Path(trace_file).open('a')as stream:stream.write(json.dumps(record)+'\n')
            finally:
                busy=False

    model.visual.forward=forward
    model.visual._causalact_staged=True
    return dict(staged_layer='model.layers.0',weight_bytes=weight_bytes,
        native_visual_forward_called_unchanged=True,CPU_or_disk_model_offload=False,
        inference_must_be_serial=True,policy_inputs_and_dtype_unchanged=True)
