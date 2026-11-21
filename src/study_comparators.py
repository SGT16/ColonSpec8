import numpy as np


def build_model(name, cfg):
    if name == "nnunet2d_supervised":
        from fewlabel_external.nnunet_adapter import build_model as build
        return build(name, cfg)
    if name != "hybridsn_supervised":
        raise ValueError(f"Unknown model {name}")
    import torch
    from fewlabel_external.common import verify_pca30
    from fewlabel_external.hybridsn_adapter import build_model as build
    arrays, _ = verify_pca30()
    weight = arrays["components"] / np.sqrt(arrays["eigenvalues"][:, None])
    bias = -(weight @ arrays["mean"])
    return build(torch.from_numpy(weight.astype(np.float32)),
                 torch.from_numpy(bias.astype(np.float32)))


def fit_gm(data, out):
    from fewlabel_gmtssa.model import TrainOnlyGMTSSA
    from fewlabel_gmtssa.tensor_ssa import iter_tiles, transform_tile
    from fewlabel.runtime import sha, write_json
    model = TrainOnlyGMTSSA(n_bands=351, rank=20)
    for image in data.train_ids:
        height, width = data.image_shape(image)
        labels = data.masks[image].reshape(-1)
        indices = np.flatnonzero(labels >= 0)
        ys, xs = np.divmod(indices, width)
        features = np.empty((len(indices), 351), dtype=np.float64)
        covered = np.zeros(len(indices), dtype=bool)
        source = np.load(data.local_cubes[image], mmap_mode="r", allow_pickle=False)
        try:
            for y0, y1, x0, x1 in iter_tiles(height, width, (230, 258)):
                values = np.array(source[y0:y1, x0:x1], dtype=np.float64, copy=True)
                values -= data.mean[:, 0, 0]
                values /= data.std[:, 0, 0]
                tile = transform_tile(values, u=5, neighbors=60, chunk_pixels=256)
                selected = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
                features[selected] = tile[ys[selected]-y0, xs[selected]-x0]
                covered[selected] = True
        finally:
            source._mmap.close()
        if not covered.all():
            raise ValueError("Features missing for revealed labels")
        for start in range(0, len(indices), 2048):
            group = slice(start, start + 2048)
            model.update(features[group], labels[indices[group]], split="TRAIN")
    target = out / "model.npz"
    model.finalize().save(target)
    write_json(out / "result.json", dict(model="gmtssa_train_only", seed=data.seed,
               status="COMPLETED", checkpoint_sha256=sha(target)))
