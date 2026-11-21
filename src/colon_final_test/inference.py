import time
import numpy as np

def infer_image(student,provider,image_id,cfg,device='cuda',progress=None):
    import torch
    from fewlabel.training import _tiles
    h,w=provider.image_shape(image_id);size=int(cfg['model']['patch_size']);stride=int(cfg['evaluation']['tile_stride'])
    summed=np.zeros((h,w,2),np.float32);coverage=np.zeros((h,w),np.uint16)
    locations=[(y,x) for y in _tiles(h,size,stride) for x in _tiles(w,size,stride)]
    last=time.monotonic()
    with torch.no_grad():
        for offset in range(0,len(locations),2):
            loc=locations[offset:offset+2]
            patches=torch.stack([torch.as_tensor(provider.get_patch(image_id,y,x,size),dtype=torch.float32) for y,x in loc])
            probs=student(patches.to(device))['logits'].float().softmax(1).permute(0,2,3,1).cpu().numpy()
            if not np.isfinite(probs).all(): raise FloatingPointError('Nonfinite TEST output')
            for (y,x),p in zip(loc,probs):
                dh,dw=min(size,h-y),min(size,w-x)
                summed[y:y+dh,x:x+dw]+=p[:dh,:dw];coverage[y:y+dh,x:x+dw]+=1
            if progress and (time.monotonic()-last>10 or offset+2>=len(locations)):
                progress(tiles_done=min(offset+2,len(locations)),tiles_total=len(locations));last=time.monotonic()
    if not np.all(coverage>0): raise RuntimeError('Uncovered TEST pixels')
    result=summed/coverage[:,:,None]
    result/=result.sum(axis=2,keepdims=True)
    return result
