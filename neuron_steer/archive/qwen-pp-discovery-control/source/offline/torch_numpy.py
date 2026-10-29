"""Read this study's own CPU tensor archives without a local PyTorch install.
Restricted pickle classes; raw storage bounds and byte order are checked.
"""
import collections,io,pickle,zipfile
import numpy as np


def load(path):
    with zipfile.ZipFile(path) as archive:
        prefix=next(n.rsplit('/',1)[0] for n in archive.namelist() if n.endswith('/data.pkl'))
        assert archive.read(prefix+'/byteorder')==b'little'
        storage_cache={}
        def tensor(storage,offset,size,stride,*rest):
            arr=storage;offset=int(offset);size=tuple(int(x) for x in size);stride=tuple(int(x) for x in stride)
            assert len(size)==len(stride) and all(x>=0 for x in size) and all(x>=0 for x in stride)
            final=offset+sum((n-1)*s for n,s in zip(size,stride) if n)
            assert offset>=0 and final<len(arr),(offset,size,stride,len(arr))
            return np.lib.stride_tricks.as_strided(arr[offset:],shape=size,strides=tuple(s*arr.dtype.itemsize for s in stride),writeable=False)
        class Reader(pickle.Unpickler):
            def find_class(self,module,name):
                if module=='collections' and name=='OrderedDict':return collections.OrderedDict
                if module=='torch._utils' and name in ['_rebuild_tensor_v2','_rebuild_tensor']:return tensor
                if module=='torch' and name in ['FloatStorage','DoubleStorage','LongStorage','IntStorage','HalfStorage','BoolStorage','ByteStorage']:
                    return {'FloatStorage':'<f4','DoubleStorage':'<f8','LongStorage':'<i8','IntStorage':'<i4','HalfStorage':'<f2','BoolStorage':'?','ByteStorage':'u1'}[name]
                raise pickle.UnpicklingError((module,name))
            def persistent_load(self,pid):
                kind,dtype,key,location,count=pid;assert kind=='storage' and location=='cpu'
                if key not in storage_cache:
                    raw=archive.read(prefix+'/data/'+str(key));assert len(raw)==count*np.dtype(dtype).itemsize
                    storage_cache[key]=np.frombuffer(raw,dtype=dtype)
                return storage_cache[key]
        return Reader(io.BytesIO(archive.read(prefix+'/data.pkl'))).load()
