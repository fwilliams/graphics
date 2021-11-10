import argparse
import bisect
import os
import time

import h5py
import numpy as np
import point_cloud_utils as pcu
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

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


def category_names_to_indices(subcategories, dataset_root, mode):
    category_names = ['chair', 'cabinet', 'airplane', 'watercraft', 'telephone', 'lamp', 'bench', 'sofa',
                      'rifle', 'car', 'table', 'loudspeaker', 'display']
    category_numbers = [category_name_to_number[n] for n in category_names]
    h5paths = [os.path.join(dataset_root, cat + "_" + mode + ".h5") for cat in category_numbers]
    category_end_indices = [0]
    for h5path in h5paths:
        with h5py.File(h5path, "r") as h5f:
            category_end_indices.append(h5f["surface_points/points"].shape[0] + category_end_indices[-1])
    name_to_indices = {category_names[i]: (category_end_indices[i], category_end_indices[i+1])
                       for i in range(len(category_names))}
    indices = []
    for category in subcategories:
        indices.append(np.arange(*name_to_indices[category]))
    return np.concatenate(indices)


def shapenet_shapes(dataset_root, input_points_h5, mode, start_from=0, shuffle=False):

    assert not shuffle
    # category_names = ['chair', 'cabinet', 'airplane', 'watercraft', 'telephone', 'lamp', 'bench', 'sofa',
    #                   'rifle', 'car', 'table', 'loudspeaker', 'display']
    # category_names = ['airplane','lamp','display','rifle','chair','cabinet']
    category_names = ['bench', 'car', 'loudspeaker', 'sofa', 'table', 'telephone', 'watercraft']
    category_numbers = [category_name_to_number[n] for n in category_names]
    h5paths = [os.path.join(dataset_root, cat + "_" + mode + ".h5") for cat in category_numbers]
    category_end_indices = [0]
    for h5path in h5paths:
        with h5py.File(h5path, "r") as h5f:
            category_end_indices.append(h5f["surface_points/points"].shape[0] + category_end_indices[-1])

    total_num_shapes = category_end_indices[-1]
    h5f = [h5py.File(h5path, "r") for h5path in h5paths]
    inptsf = h5py.File(input_points_h5, "r")
    assert total_num_shapes == len(inptsf["input_points/points"])

    # index_map = category_names_to_indices(category_names, dataset_root, mode)
    # assert len(index_map) == total_num_shapes
    indexes = np.random.permutation(total_num_shapes)[start_from:] if shuffle else range(start_from, total_num_shapes)
    for idx in indexes:
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
            "vol_occs": vol_occ,
            "num_shapes": total_num_shapes
        }


def intersection_over_union(pred, target):
    """
    Compute the intersection over union between two indicator sets
    :param pred: A boolean tensor of shape [*, N] representing an indicator set of size N.
                 True values indicate points *inside* the set, and false indicate outside value.
    :param target: A boolean tensor of shape [*, N] representing an indicator set of size N.
                   True values indicate points *inside* the set, and false indicate outside value.
    :return: A tensor of shape [*] storing the intersection over union
    """
    intersection = np.logical_and(pred, target).sum(-1)
    union = np.logical_or(pred, target).sum(-1)
    return intersection / union


def chamfer_distance(pc_p, pc_t, return_index=False, return_individual=False, p_norm=2):
    """
    Computes the Chamfer distance between the predicted (p) and target (t) point cloud
    :param pc_p: predicted point cloud [n,3]
    :param pc_t: target point cloud [m,3]
    :param return_index: If true, indices of the closest points will be returned
    :param return_individual: If true, onesided CD will be returned instead of te twosided one
    :param p_norm: p of the p-distance
    :return d_p2t: one sided CD distance of the predicted points to the target ones [n]
    :return d_t2p: one sided CD distance of the target points to the predicted ones [m]
    :return nn_idx_p2t: indices of the target points that are the closest to the predicted ones [n]
    :return nn_idx_t2p: indices of the predicted points that are the closest to the target ones [m]
    :return cham_dist: two sided mean chamfer distance [1]
    """
    tree_p = cKDTree(pc_p)
    tree_t = cKDTree(pc_t)
    _, nn_idx_t2p = tree_p.query(pc_t, k=1, p=p_norm)
    _, nn_idx_p2t = tree_t.query(pc_p, k=1, p=p_norm)

    # Compute the distances
    d_t2p = np.linalg.norm(pc_p[nn_idx_t2p] - pc_t, axis=-1, ord=p_norm)
    d_p2t = np.linalg.norm(pc_t[nn_idx_p2t] - pc_p, axis=-1, ord=p_norm)
    cham_dist = d_t2p.mean() + d_p2t.mean()

    # Handle the return parameters
    if return_index:
        if return_individual:
            return d_p2t, d_t2p, nn_idx_p2t, nn_idx_t2p
        else:
            return cham_dist, nn_idx_p2t, nn_idx_t2p

    if return_individual:
        return d_p2t, d_t2p
    else:
        return cham_dist


def one_sided_hausdorff_distance(x, y, return_index=False, p_norm=2):
    tree_x = cKDTree(x)
    d_y_to_x, i_y_to_x = tree_x.query(y, k=1, p=p_norm)
    d1 = np.linalg.norm(x[i_y_to_x] - y, axis=-1, ord=p_norm)
    max_idx = np.argmax(d1)
    hausdorff = d1[max_idx]

    if return_index:
        return hausdorff, i_y_to_x[max_idx], max_idx
    return hausdorff


def hausdorff_distance(x, y, return_index=False, p_norm=2):
    hausdorff_x_to_y, idx_x1, idx_y1 = one_sided_hausdorff_distance(x, y, return_index=True, p_norm=p_norm)
    hausdorff_y_to_x, idx_y2, idx_x2 = one_sided_hausdorff_distance(y, x, return_index=True, p_norm=p_norm)

    hausdorff = max(hausdorff_x_to_y, hausdorff_y_to_x)
    if return_index and hausdorff_x_to_y > hausdorff_y_to_x:
        return hausdorff, idx_x1, idx_y1
    elif return_index and hausdorff_x_to_y <= hausdorff_y_to_x:
        return hausdorff, idx_x2, idx_y2
    return hausdorff


def normals_similarity(normals_pre, normals_tgt, idx):
    """
    Compute the normal vector similarity metric
    :param normals_pre: Numpy array of the predicted normal vectors [n, 3]
    :param normals_tgt: Numpy array of the target (GT) normal vectors [n, 3]
    :param idx: indices of the closest points in the target point cloud (source -> target) [n]
    :return norm_similarity: similarity measure of the normal vectors [n]
    """
    # Normalize the normal vectors to unit length
    normals_pre = normals_pre / np.linalg.norm(normals_pre, axis=-1, keepdims=True)
    normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

    normals_dot_product = (normals_tgt[idx] * normals_pre).sum(axis=-1)

    # Handle normals that point into wrong direction gracefully
    # (mostly due to mehtod not caring about this in generation)
    norm_similarity = np.abs(normals_dot_product)

    return norm_similarity.mean()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("dataset_root", type=str)
    argparser.add_argument("input_points_h5", type=str)
    argparser.add_argument("output_path", type=str)
    argparser.add_argument("--mode", type=str, default="test")
    argparser.add_argument("--iters", type=int, default=10_000)
    argparser.add_argument("--part-size", type=float, default=0.33)
    argparser.add_argument("--absolute", action="store_true")
    argparser.add_argument("--save-every", type=int, default=20)
    argparser.add_argument("--shuffle", action="store_true")
    argparser.add_argument("--resume", action="store_true")
    cmd_args = argparser.parse_args()

    if not os.path.exists(cmd_args.output_path) and not cmd_args.resume:
        os.makedirs(cmd_args.output_path)
        iou_losses = []
        chamfer_l2_losses = []
        hausdorff_losses = []
        normal_consistency_losses = []
        runtimes = []
        start_from = 0
    elif cmd_args.resume:
        stats = np.load(os.path.join(cmd_args.output_path, "stats.npz"), allow_pickle=True)
        iou_losses = list(stats['iou_loss'])
        chamfer_l2_losses = list(stats['chamfer_loss_l2'])
        hausdorff_losses = list(stats['hausdorff_loss'])
        normal_consistency_losses = list(stats['norm_similarities'])
        runtimes = list(stats['runtimes'])
        assert len(iou_losses) == len(chamfer_l2_losses) == len(hausdorff_losses) == \
               len(normal_consistency_losses) == len(runtimes)
        start_from = len(iou_losses)
    else:
        assert False, "Unwilling to overwrite existing data"

    for idx, shape in enumerate(shapenet_shapes(cmd_args.dataset_root, cmd_args.input_points_h5,
                                                cmd_args.mode, start_from=start_from, shuffle=cmd_args.shuffle)):
        v_in, n_in = shape['in_points'], shape['in_normals']
        for fname in ["in_pts.ply", "in_pts.reconstruct.ply", "in_pts.reconstruct.ply.npz"]:
            if os.path.exists(fname):
                os.remove(fname)
        pcu.save_mesh_vn("in_pts.ply", v_in, n_in)

        res_per_part = 32

        min_bb = np.min(np.max(v_in, axis=0) - np.min(v_in, axis=0))
        part_size = cmd_args.part_size if cmd_args.absolute else cmd_args.part_size * min_bb
        assert cmd_args.part_size > 0
        print(f"part_size = {part_size}")

        start_time = time.time()
        os.system(f"python reconstruct_geometry.py --input_ply in_pts.ply "
                  f"--part_size={part_size} --npoints=2048 --steps={cmd_args.iters} --res_per_part={res_per_part}")
        end_time = time.time()
        runtime = end_time - start_time

        v, f = pcu.load_mesh_vf("in_pts.reconstruct.ply")
        n = pcu.estimate_mesh_normals(v, f)

        grid_data = np.load("in_pts.reconstruct.ply.npz")
        grid = grid_data["grid"]

        # Sample reconstructed grid
        eps = 1e-6
        grid_shape = grid_data["grid_shape"]
        s = ((np.array(grid_shape) - 1) / 2.0).astype(np.int)
        xmin, xmax = grid_data["xmin"], grid_data["xmin"] + s * part_size
        ll = tuple([np.linspace(xmin[i] + eps, xmax[i] - eps, res_per_part * s[i]) for i in range(3)])
        interpolator = RegularGridInterpolator(ll, grid, bounds_error=False, fill_value=1.0)
        vol_pts = shape['vol_points']

        # IoU
        gt_occ = shape['vol_occs'] <= 0.0
        pred_occ = interpolator(vol_pts) <= 0.0
        fid, bc = pcu.sample_mesh_random(v, f, 100_000)
        gt_surf_pts = shape['surf_points']
        pred_surf_pts = pcu.interpolate_barycentric_coords(f, fid, bc, v)
        gt_surf_nms = shape['surf_points']
        pred_surf_nms = pcu.interpolate_barycentric_coords(f, fid, bc, n)

        cd_p2t, cd_t2p, nn_idx_p2t, nn_idx_t2p = chamfer_distance(pred_surf_pts, gt_surf_pts,
                                                                  return_index=True, return_individual=True)
        chamfer_distance_l2 = 0.5 * (cd_p2t.mean() + cd_t2p.mean())
        hausdorff_distance_l2 = hausdorff_distance(pred_surf_pts, gt_surf_pts)
        normal_similarity = max(normals_similarity(pred_surf_nms, gt_surf_nms, nn_idx_p2t),
                                normals_similarity(-pred_surf_nms, gt_surf_nms, nn_idx_p2t))
        iou = intersection_over_union(pred_occ, gt_occ)

        iou_losses.append(iou)
        chamfer_l2_losses.append(chamfer_distance_l2)
        hausdorff_losses.append(hausdorff_distance)
        normal_consistency_losses.append(normal_similarity)
        runtimes.append(runtime)
        np.savez(os.path.join(cmd_args.output_path, "stats.npz"),
                 iou_loss=np.array(iou_losses),
                 chamfer_loss_l2=np.array(chamfer_l2_losses),
                 hausdorff_loss=np.array(hausdorff_losses),
                 norm_similarities=np.array(normal_consistency_losses),
                 runtime=np.array(runtimes))
        real_idx = idx + start_from
        if real_idx % cmd_args.save_every == 0:
            print(f"Saving at iteration {real_idx}")
            pcu.save_mesh_vfn(os.path.join(cmd_args.output_path, f"recon_{real_idx}.ply"), v, f, n)
            pcu.save_mesh_vn(os.path.join(cmd_args.output_path, f"pts_{real_idx}.ply"), v_in, n_in)
            pcu.save_mesh_v(os.path.join(cmd_args.output_path, f"chamfer_pts_{real_idx}.ply"), gt_surf_pts)

        print(f"{real_idx}/{shape['num_shapes']} {runtime}s: IoU: {iou}, Chamfer L2: {chamfer_distance_l2}, "
              f"Hausdorff Distance: {hausdorff_distance_l2}, Normal Consistency: {normal_similarity}")


if __name__ == "__main__":
    main()
    
