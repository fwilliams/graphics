import h5py
import os
import argparse
import bisect
import numpy as np
import point_cloud_utils as pcu
from scipy.interpolate import RegularGridInterpolator


def shapenet_shapes(dataset_root, input_pts_root, mode):
    category_name_to_number = {
        "airplane": "02691156",
        "bench": "02828884",
        "cabinet": "02933112",
        "car": "02958343",
        "chair": "03001627",
        "display": "03211117",
        "lamp": "03636649",
        "loudspeaker": "03691459",
        "rifle": "04090263",
        "sofa": "04256520",
        "table": "04379243",
        "telephone": "04401088",
        "watercraft": "04530566",
    }

    category_names = ['chair', 'cabinet', 'airplane', 'watercraft', 'telephone', 'lamp', 'bench', 'sofa',
                      'rifle', 'car', 'table', 'loudspeaker', 'display']
    category_numbers = [category_name_to_number[n] for n in category_names]
    h5paths = [os.path.join(dataset_root, cat + "_" + mode + ".h5") for cat in category_numbers]
    category_end_indices = [0]
    for h5path in h5paths:
        with h5py.File(h5path, "r") as h5f:
            category_end_indices.append(h5f["surface_points/points"].shape[0] + category_end_indices[-1])

    total_num_shapes = category_end_indices[-1]
    h5f = [h5py.File(h5path, "r") for h5path in h5paths]
    inptsf = h5py.File(input_pts_root, "r")

    for idx in range(total_num_shapes):
        # retrieve the file idx and shape idx within that file
        file_idx = bisect.bisect_right(category_end_indices, idx) - 1
        shape_idx = idx - category_end_indices[file_idx]

        # Hardcoded transformation that makes all points live in [-1, 1]^3
        translate, scale = 0.0, 2.0

        in_pts = inptsf["input_points/points"][idx].astype(np.float32)
        in_pts = scale * (in_pts + translate)

        in_nms = inptsf["input_points/normals"][idx].astype(np.float32)

        surf_pts = h5f[file_idx]["surface_points/points"][shape_idx].astype(np.float32)
        surf_pts = scale * (surf_pts + translate)

        surf_nms = h5f[file_idx]["surface_points/normals"][shape_idx].astype(np.float32)

        vol_pts = h5f[file_idx]["volume_points/points"][shape_idx].astype(np.float32)
        vol_pts = scale * (vol_pts + translate)

        vol_occ = -2.0 * (
                np.unpackbits(h5f[file_idx]["volume_points/occupancies"][shape_idx]).astype(np.float32) - 0.5)

        yield {
            'shape_id': idx,
            'scale': scale,
            "in_points": in_pts,
            "in_normals": in_nms,
            "surf_points": surf_pts,
            "surf_normals": surf_nms,
            "vol_points": vol_pts,
            "vol_occs": vol_occ
        }


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("dataset_root", type=str)
    argparser.add_argument("input_points_root", type=str)
    argparser.add_argument("output_path", type=str)
    argparser.add_argument("--mode", type=str, default="test")
    argparser.add_argument("--iters", type=int, default=10_000)
    cmd_args = argparser.parse_args()

    for idx, shape in enumerate(shapenet_shapes(cmd_args.dataset_root, cmd_args.input_points_root, cmd_args.mode)):
        v_in, n_in = shape['in_points'], shape['in_normals']
        if os.path.exists("in_pts.ply"):
            os.remove("in_pts.ply")
        pcu.save_mesh_vn("in_pts.ply", v_in, n_in)

        res_per_part = 32
        part_size = 0.15
        os.system(f"python reconstruct_geometry.py --input_ply in_pts.ply "
                  f"--part_size={part_size} --npoints=2048 --steps={cmd_args.iters} --res_per_part={res_per_part}")

        v, f = pcu.load_mesh_vf("in_pts.reconstruct.ply")

        grid_data = np.load("in_pts.reconstruct.ply.npz")
        grid = grid_data["grid"]
        xmin, xmax = grid_data["xmin"], grid_data["xmax"]
        grid_shape = grid_data["grid_shape"]

        # setup grid
        eps = 1e-6
        s = ((np.array(grid_shape) - 1) / 2.0).astype(np.int)
        print("my s", s)
        print("my xmin/xmax", xmin, xmax)
        ll = tuple([np.linspace(xmin[i] + eps, xmax[i] - eps, res_per_part * s[i]) for i in range(3)])
        # xx, yy, zz = np.meshgrid(ll[0], ll[1], ll[2], indexing='ij')
        # print(xx.shape, yy.shape, zz.shape, grid.shape)
        # output_grid = np.ones([res_per_part * s[0], res_per_part * s[1],  res_per_part * s[2]],
        #                       dtype=np.float32).reshape(-1)
        interpolator = RegularGridInterpolator(ll, grid, bounds_error=False, fill_value=1.0)
        vol_pts = shape['vol_points']
        pred_occ = interpolator(vol_pts) <= 0.0
        gt_occ = shape['vol_occs'] <= 0.0

        print("my l", [(l.min(), l.max()) for l in ll])
        print("vol_pts min/max", vol_pts.min(0), vol_pts.max(0))
        print("xmin/xmax", xmin, xmax)
        print("vmin/vmax", v.min(0), v.max(0))
        np.savez("debugme", vol_pts=vol_pts, pred_occ=pred_occ, gt_occ=gt_occ)

        assert False


if __name__ == "__main__":
    main()
    
