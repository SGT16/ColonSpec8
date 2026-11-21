from pathlib import Path
import numpy as np
from fewlabel.runtime import read_json as read, sha, write_json, now

class TestImages:
    def __init__(self, root):
        self.cfg=read(root / "configs/test_protocol.json")
        self.ids=self.cfg["test_ids"]
        self.raw=root / "data/raw"
        self.sources={name:dict(size_bytes=values[0], sha256=values[1]) for name,values in read(root / "configs/data_checksums.json").items()}
        normal=read(root / self.cfg["normalization_path"])
        self.mean=np.asarray(normal["mean"],dtype=np.float32)
        self.std=np.asarray(normal["std"],dtype=np.float32)
        if self.mean.shape!=(351,) or self.std.shape!=(351,) or not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)) or np.any(self.std<=0):
            raise ValueError("Invalid fixed TRAIN scaler")
        self.cache=root / "data/processed/test_cache"
        self.cache.mkdir(parents=True,exist_ok=True)
        self.active=self.cube=None
        self.verify_raw(self.cfg["test_gt"])
        from scipy.io import loadmat
        gt=loadmat(self.raw/self.cfg["test_gt"],variable_names=["GT_test"])["GT_test"]
        if list(gt.shape)!=self.cfg["expected_gt_shape"] or not np.isin(gt,[0,1]).all():
            raise ValueError("Invalid dense test mask")
        self.truth=gt.astype(np.int8)

    def verify_raw(self,name):
        item=self.sources[name];p=self.raw/name
        if p.stat().st_size!=item['size_bytes'] or sha(p)!=item['sha256']:
            raise RuntimeError('Raw TEST identity mismatch: '+name)

    def prepare_image(self,image_id,progress=None):
        if image_id not in self.ids: raise ValueError('Non-frozen TEST ID')
        target=self.cache/(image_id+'.hwc.npy');meta=target.with_suffix('.json')
        source=self.sources[image_id+'.h5']
        if target.exists() or meta.exists():
            if not target.exists() or not meta.exists(): raise RuntimeError('Partial TEST cache; review required')
            info=read(meta)
            if info['source_sha256']!=source['sha256'] or sha(target)!=info['sha256']:
                raise RuntimeError('TEST cache identity changed')
            return target
        self.verify_raw(image_id+'.h5')
        import h5py
        pending=target.with_suffix('.pending')
        if pending.exists(): raise RuntimeError('Interrupted TEST cache; no automatic retry')
        with h5py.File(self.raw/(image_id+'.h5'),'r') as f:
            ds=f['/img']
            if list(ds.shape)!=self.cfg['expected_h5_shape']: raise ValueError('Unexpected TEST H5 shape')
            cube=np.lib.format.open_memmap(pending,mode='w+',dtype=np.float32,shape=(1384,1035,351))
            try:
                for y in range(0,1384,32):

                    slab=np.asarray(ds[:,:,y:min(y+32,1384)],dtype=np.float64)
                    if not np.isfinite(slab).all(): raise ValueError('Nonfinite raw TEST input')
                    chunk=np.ascontiguousarray(slab.transpose(2,1,0),dtype=np.float32)
                    if not np.isfinite(chunk).all(): raise ValueError('Float32 TEST overflow')
                    cube[y:y+len(chunk)]=chunk
                    if progress: progress(stage='test_image_preparation',image_id=image_id,row=min(y+32,1384))
                cube.flush()
            finally: cube._mmap.close()
        pending.replace(target)
        write_json(meta,dict(image_id=image_id,source_sha256=source['sha256'],sha256=sha(target),
            shape=[1384,1035,351],dtype='float32',preprocessing_fitted=False,created_at=now()))
        return target

    def open_image(self,image_id):
        self.close()
        if image_id not in self.ids: raise ValueError('Non-frozen TEST ID')
        self.cube=np.load(self.cache/(image_id+'.hwc.npy'),mmap_mode='r',allow_pickle=False)
        if self.cube.shape!=(1384,1035,351) or self.cube.dtype!=np.float32: raise ValueError('TEST cache shape/dtype')
        self.active=image_id

    def close(self):
        if self.cube is not None: self.cube._mmap.close()
        self.cube=None;self.active=None

    def image_shape(self,image_id):
        if image_id not in self.ids: raise ValueError('Unknown TEST image')
        return (1384,1035)

    def labels(self,image_id): return self.truth[:,:,self.ids.index(image_id)]

    def get_patch(self,image_id,y,x,size):
        if self.active!=image_id: raise RuntimeError('Image not opened')
        patch=np.array(self.cube[y:y+size,x:x+size,:].transpose(2,0,1),dtype=np.float32,copy=True)
        patch-=self.mean[:,None,None];patch/=self.std[:,None,None]
        out=np.zeros((351,size,size),np.float32);out[:,:patch.shape[1],:patch.shape[2]]=patch
        return out

    def get_gm_tile(self,image_id,bounds):
        if self.active!=image_id: raise RuntimeError('Image not opened')
        y0,y1,x0,x1=bounds
        value=np.array(self.cube[y0:y1,x0:x1,:],dtype=np.float64,copy=True)
        value-=self.mean.astype(np.float64)[None,None,:]
        value/=self.std.astype(np.float64)[None,None,:]
        return value
