import contextlib
import copy
from pathlib import Path
import tempfile
import types
import unittest

from uitars_cuda_staging import stage_first_decoder_during_vision


class Tensor:
    dtype='fp16'
    value=7
    def __init__(self):self.device=types.SimpleNamespace(type='cuda',index=0)
    def numel(self):return 10
    def element_size(self):return 2
    def detach(self):return self
    def cpu(self):return copy.deepcopy(self)
    def clone(self):return copy.deepcopy(self)


class Layer:
    def __init__(self):self.tensor=Tensor();self.moves=[]
    def parameters(self):return iter([self.tensor])
    def state_dict(self):return {'weight':self.tensor}
    def to(self,device):self.moves.append(device);self.tensor.device.index=int(device[-1]);return self


def fixture(fail=False,free=1024**3):
    layer=Layer();marker=object();seen=[]
    def native(value):
        seen.append(layer.tensor.device.index)
        if fail:raise ValueError('original visual error')
        return marker
    model=types.SimpleNamespace(training=False,config=types.SimpleNamespace(model_type='qwen2_5_vl'),
        model=types.SimpleNamespace(layers=[layer]*28),visual=types.SimpleNamespace(forward=native),
        hf_device_map={'visual':0,'model.layers.0':0,'lm_head':1})
    torch=types.SimpleNamespace(float16='fp16',equal=lambda a,b:a.value==b.value,
        cuda=types.SimpleNamespace(device=lambda i:contextlib.nullcontext(),empty_cache=lambda:None,
            synchronize=lambda i:None,mem_get_info=lambda i:(free,2*1024**3)))
    return model,torch,layer,marker,seen


class StagingTests(unittest.TestCase):
    def test_preserves_output_and_values_and_restores_before_decode(self):
        with tempfile.TemporaryDirectory()as folder:
            m,t,layer,marker,seen=fixture()
            stage_first_decoder_during_vision(m,t,Path(folder)/'trace.jsonl')
            self.assertIs(m.visual.forward('input'),marker)
            self.assertEqual(seen,[1])
            self.assertEqual(layer.moves,['cuda:1','cuda:0'])
            self.assertEqual(layer.tensor.device.index,0)
            self.assertEqual(layer.tensor.value,7)
            self.assertIs(m.visual.forward('next input'),marker)

    def test_native_failure_still_restores_weights(self):
        with tempfile.TemporaryDirectory()as folder:
            m,t,layer,_,_=fixture(fail=True)
            stage_first_decoder_during_vision(m,t,Path(folder)/'trace.jsonl')
            with self.assertRaisesRegex(ValueError,'original visual error'):m.visual.forward('input')
            self.assertEqual(layer.tensor.device.index,0)

    def test_insufficient_memory_does_not_call_native_vision(self):
        with tempfile.TemporaryDirectory()as folder:
            m,t,layer,_,seen=fixture(free=1)
            stage_first_decoder_during_vision(m,t,Path(folder)/'trace.jsonl')
            with self.assertRaisesRegex(RuntimeError,'No destination memory'):m.visual.forward('input')
            self.assertFalse(seen)
            self.assertEqual(layer.tensor.device.index,0)

    def test_rejects_training_offload_wrong_dtype_and_duplicate_install(self):
        with tempfile.TemporaryDirectory()as folder:
            trace=Path(folder)/'trace.jsonl'
            for change in ['training','offload','dtype']:
                m,t,layer,_,_=fixture()
                if change=='training':m.training=True
                if change=='offload':m.hf_device_map['lm_head']='cpu'
                if change=='dtype':layer.tensor.dtype='fp32'
                with self.assertRaises(RuntimeError):stage_first_decoder_during_vision(m,t,trace)
            m,t,_,_,_=fixture();stage_first_decoder_during_vision(m,t,trace)
            with self.assertRaises(RuntimeError):stage_first_decoder_during_vision(m,t,trace)


if __name__=='__main__':unittest.main()
