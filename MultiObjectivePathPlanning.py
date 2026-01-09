from Problem import Problem
from Evolution import Evolution
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt, gaussian_filter, sobel
from skimage.segmentation import find_boundaries

def trajectory_length(x1, y1, z1, x2, y2, z2):
    if (x1, y1, z1) not in surface_voxel_to_index:
        raise ValueError(f"Illegal coordinates:({x1}, {y1}, {z1})")
    if (x2, y2, z2) not in tumor_voxel_to_index:
        raise ValueError(f"Illegal coordinates:({x2}, {y2}, {z2})")

    p1_world = surface_coords_world[surface_voxel_to_index[(x1, y1, z1)]]
    p2_world = tumor_coords_world[tumor_voxel_to_index[(x2, y2, z2)]]

    return np.linalg.norm(p2_world - p1_world)

def entrance_angle(x1, y1, z1, x2, y2, z2):
    if (x1, y1, z1) not in surface_voxel_to_index:
        raise ValueError(f"Illegal coordinates:({x1}, {y1}, {z1})")
    if (x2, y2, z2) not in tumor_voxel_to_index:
        raise ValueError(f"Illegal coordinates:({x2}, {y2}, {z2})")

    normal_world = surface_normals_world[surface_voxel_to_index[(x1, y1, z1)]]
    p1_world = surface_coords_world[surface_voxel_to_index[(x1, y1, z1)]]
    p2_world = tumor_coords_world[tumor_voxel_to_index[(x2, y2, z2)]]
    direction_world = p2_world - p1_world
    direction_norm = np.linalg.norm(direction_world)
    cos_theta = np.dot(direction_world, normal_world) / direction_norm
    return -abs(cos_theta)


def risk_cost(x1, y1, z1, x2, y2, z2):

    points = []
    dx, dy, dz = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
    xs, ys, zs = (1 if x2 > x1 else -1), (1 if y2 > y1 else -1), (1 if z2 > z1 else -1)
    if dx >= dy and dx >= dz:
        err_1, err_2 = 2 * dy - dx, 2 * dz - dx
        while x1 != x2:
            points.append((x1, y1, z1))
            if err_1 > 0:
                y1 += ys
                err_1 -= 2 * dx
            if err_2 > 0:
                z1 += zs
                err_2 -= 2 * dx
            err_1 += 2 * dy
            err_2 += 2 * dz
            x1 += xs

    elif dy >= dx and dy >= dz:
        err_1, err_2 = 2 * dx - dy, 2 * dz - dy
        while y1 != y2:
            points.append((x1, y1, z1))
            if err_1 > 0:
                x1 += xs
                err_1 -= 2 * dy
            if err_2 > 0:
                z1 += zs
                err_2 -= 2 * dy
            err_1 += 2 * dx
            err_2 += 2 * dz
            y1 += ys

    else:
        err_1, err_2 = 2 * dy - dz, 2 * dx - dz
        while z1 != z2:
            points.append((x1, y1, z1))
            if err_1 > 0:
                y1 += ys
                err_1 -= 2 * dz
            if err_2 > 0:
                x1 += xs
                err_2 -= 2 * dz
            err_1 += 2 * dy
            err_2 += 2 * dx
            z1 += zs

    points.append((x1, y1, z1))

    total_risk = sum(risk_map[i, j, k] for i, j, k in points
                     if 0 <= i < risk_map.shape[0] and
                        0 <= j < risk_map.shape[1] and
                        0 <= k < risk_map.shape[2])

    return total_risk

risk_map = np.load('riskmap.npy')
head_img = nib.load('Head_label.nii')
tumor_img = nib.load('Tumor-label.nii')

tumor_mask = tumor_img.get_fdata().astype(bool)
voxel_spacing_tumor = tumor_img.header.get_zooms()[:3]
affine_tumor = tumor_img.affine
distance_inside = distance_transform_edt(tumor_mask, sampling=voxel_spacing_tumor)
tumor_eroded_mask = distance_inside >= 1.0
tumor_coords = np.argwhere(tumor_eroded_mask)
tumor_coords_world = nib.affines.apply_affine(affine_tumor, tumor_coords)

head_mask = head_img.get_fdata().astype(bool)
voxel_spacing_head = head_img.header.get_zooms()[:3]
affine_head = head_img.affine
head_mask = binary_closing(head_mask)
head_mask = binary_fill_holes(head_mask)
surface_mask = find_boundaries(head_mask, mode='outer')
surface_coords = np.argwhere(surface_mask)
surface_coords_world = nib.affines.apply_affine(affine_head, surface_coords)

surface_voxel_to_index = {(x, y, z): idx for idx, (x, y, z) in enumerate(surface_coords)}
tumor_voxel_to_index = {(x, y, z): idx for idx, (x, y, z) in enumerate(tumor_coords)}

head_mask_float = gaussian_filter(head_mask.astype(float), sigma=1.0)
grad_x_img = sobel(head_mask_float, axis=0)
grad_y_img = sobel(head_mask_float, axis=1)
grad_z_img = sobel(head_mask_float, axis=2)
norms = np.sqrt(grad_x_img**2 + grad_y_img**2 + grad_z_img**2)
norms[norms == 0] = 1e-6
normal_map = np.stack([-grad_x_img / norms, -grad_y_img / norms, -grad_z_img / norms], axis=-1)
surface_normals = normal_map[surface_mask]
R = affine_head[:3, :3]
surface_normals_world = np.dot(surface_normals, R.T)
surface_normals_world /= np.linalg.norm(surface_normals_world, axis=1, keepdims=True)

problem = Problem([trajectory_length, entrance_angle, risk_cost], surface_coords, tumor_coords)
evolution = Evolution(problem, num_of_generations=1000, num_of_individuals=200, tournament_prob=0.7)
pareto_front = evolution.evolve()

pareto_data = []
for ind in pareto_front:
    x0, y0, z0, x1, y1, z1 = ind.features
    f1 = ind.objectives[0]
    f2 = ind.objectives[1]
    f3 = ind.objectives[2]
    pareto_data.append([x0, y0, z0, x1, y1, z1, f1, f2, f3])
pareto_data = np.array(pareto_data)
np.save("ParetoFront.npy", pareto_data)







