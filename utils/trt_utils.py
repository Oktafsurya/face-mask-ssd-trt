import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
        del host_mem, device_mem

    def __repr__(self):
        return "Host:\n{}\nDevice:\n{}".format(str(self.host), str(self.device))

def _allocate_buffer(engine):
    inputs = dict()
    outputs = dict()
    bindings = list() 

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs[binding] = HostDeviceMem(host_mem, device_mem)
        else:
            outputs[binding] = HostDeviceMem(host_mem, device_mem)
            
    return inputs, outputs, bindings

def memcpy_itoh(host, input):
    for key in input:
        np.copyto(host[key].host, input[key].tensor.ravel())


def _compile_output(output, **kwargs):
    ret = dict()
    if 'inference' in kwargs:
        for out in range(len(output)):
            ret[out.name] = kwargs['inference'].get_tensor(output[out].index)
    else:
        for key in output:
            ret[key] = output[key].host
    return ret

def result_handler(inference, **kwargs):
    inputs, outputs, bindings = _allocate_buffer(inference)
    stream = cuda.Stream()
    memcpy_itoh(inputs, kwargs['tensor'])
    [cuda.memcpy_htod_async(inputs[key].device, inputs[key].host, stream) for key in inputs]
    kwargs['context'].execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(outputs[key].host, outputs[key].device, stream) for key in outputs]
    stream.synchronize()
    return _compile_output(outputs)