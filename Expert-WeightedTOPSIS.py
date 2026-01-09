import numpy as np
from scipy.ndimage import distance_transform_edt
import nibabel as nib
def load_nifti(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata()
    affine = nii.affine
    return data, affine

def calculate_min_distances(entry_voxel, target_voxel, ventricles_data, vessels_data, motor_data, affine):
    def get_distance(mask, affine):
        distance_field = distance_transform_edt(1 - mask)
        t = np.linspace(0, 1, 1000)
        x = np.round(entry_voxel[0] + t * (target_voxel[0] - entry_voxel[0])).astype(int)
        y = np.round(entry_voxel[1] + t * (target_voxel[1] - entry_voxel[1])).astype(int)
        z = np.round(entry_voxel[2] + t * (target_voxel[2] - entry_voxel[2])).astype(int)
        x = np.clip(x, 0, mask.shape[0] - 1)
        y = np.clip(y, 0, mask.shape[1] - 1)
        z = np.clip(z, 0, mask.shape[2] - 1)
        min_dist_voxel = np.min(distance_field[x, y, z])
        voxel_size = np.linalg.det(affine) ** (1 / 3)
        return min_dist_voxel * voxel_size

    ventricle_dist = get_distance(ventricles_data, affine)
    vessel_dist = get_distance(vessels_data, affine)
    motor_dist = get_distance(motor_data, affine)
    return ventricle_dist, vessel_dist, motor_dist

def topsis_decision(decision_matrix, weights, objective_types):
    matrix = np.array(decision_matrix, dtype=float)
    norm_matrix = matrix / np.sqrt(np.sum(matrix ** 2, axis=0))
    weighted_matrix = norm_matrix * np.array(weights)
    positive_ideal = np.zeros(matrix.shape[1])
    negative_ideal = np.zeros(matrix.shape[1])
    for j in range(matrix.shape[1]):
        if objective_types[j].lower() == 'benefit':
            positive_ideal[j] = np.max(weighted_matrix[:, j])
            negative_ideal[j] = np.min(weighted_matrix[:, j])
        elif objective_types[j].lower() == 'cost':
            positive_ideal[j] = np.min(weighted_matrix[:, j])
            negative_ideal[j] = np.max(weighted_matrix[:, j])
    dist_positive = np.sqrt(np.sum((weighted_matrix - positive_ideal) ** 2, axis=1))
    dist_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal) ** 2, axis=1))
    closeness = dist_negative / (dist_positive + dist_negative + 1e-9)
    sorted_indices = np.argsort(closeness)[::-1]
    return closeness, sorted_indices


if __name__ == "__main__":

    pareto_data = np.load("ParetoFront.npy.npy")
    vessels_data, vessels_affine = load_nifti("Vessels-label.nii")
    ventricles_data, ventricles_affine = load_nifti("Ventricles_label.nii")
    motor_data, motor_affine = load_nifti("M1.nii")

    features = pareto_data[:, :6]
    objectives = pareto_data[:, 6:]

    weights = [0.35, 0.2, 0.45]
    objective_types = ['cost', 'cost', 'cost']

    topsis_scores, sorted_indices = topsis_decision(objectives, weights, objective_types)

    top5_indices = sorted_indices[:5]
    top5_data = pareto_data[top5_indices]
    top5_features = top5_data[:, :6]
    top5_objectives = top5_data[:, 6:]

    print("--- TOPSIS Ranking Results ---")
    print("Rank | Features                          | PathLength/EntryAngle/RiskCost                     | C*")
    print("-" * 120)
    for rank, idx in enumerate(top5_indices, 1):
        feat = features[idx]
        orig_obj = objectives[idx]
        print(f"{rank:2d}   | "
              f"({feat[0]:5.1f}, {feat[1]:5.1f}, {feat[2]:5.1f}, {feat[3]:5.1f}, {feat[4]:5.1f}, {feat[5]:5.1f}) | "
              f"f1={orig_obj[0]:6.2f}mm | f2={orig_obj[1]:6.2f}° | f3={orig_obj[2]:5.1%}    | "
              f"Score={topsis_scores[idx]:.4f}")

    results = []
    for i in range(5):
        feat = top5_features[i]
        length = top5_objectives[i, 0]
        f2 = top5_objectives[i, 1]
        angle = np.degrees(np.arccos(np.clip(f2, 0.0, 1.0)))
        entry_point = feat[:3]
        target_point = feat[3:]
        ventricle_dist, vessel_dist, m1_dist = calculate_min_distances(entry_point, target_point, ventricles_data, vessels_data,
                                                              motor_data, vessels_affine,)
        results.append([length, angle, ventricle_dist, vessel_dist, m1_dist])

    print("\nPath | PathLength(mm) | EntryAngle(°) | Dventricles(mm) | Dvessels(mm)| DM1(mm)")
    print("-" * 80)
    for i, res in enumerate(results):
        print(f"{i + 1:2d}  | {res[0]:12.2f} | {res[1]:12.2f} | {res[2]:15.2f} | {res[3]:15.2f}| {res[4]:15.2f}")

