import numpy as np
import nibabel as nib
from scipy.spatial import KDTree

def load_nii_image(nii_path):
    nii_image = nib.load(nii_path)
    data = nii_image.get_fdata()
    affine = nii_image.affine
    voxel_spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    return data, voxel_spacing, affine

class RiskMapGenerator:
    def __init__(self, structures, risk_values, voxel_spacing, N=5, sigma=5, d_max=20.0):
        self.structures = structures
        self.risk_values = risk_values
        self.N = N
        self.sigma = sigma
        self.d_max = d_max
        self.voxel_spacing = voxel_spacing
        self.spacing_norm = float(np.linalg.norm(voxel_spacing))
        self.kdtrees = []
        for struct in self.structures:
            coords = np.array(np.where(struct > 0)).T
            self.kdtrees.append(KDTree(coords))

    def generate_risk_map(self, original_data, batch_size=50000):
        risk_map = np.full_like(original_data, -1.0, dtype=np.float32)
        nonzero_coords = np.array(np.where(original_data > 0)).T
        total_risks = np.zeros(nonzero_coords.shape[0], dtype=np.float32)

        for struct_idx, tree in enumerate(self.kdtrees):
            risk_value = self.risk_values[struct_idx]

            for i in range(0, nonzero_coords.shape[0], batch_size):
                batch_voxels = nonzero_coords[i:i+batch_size]
                dists, _ = tree.query(batch_voxels, k=self.N, workers=-1)

                dists_phys = dists * self.spacing_norm
                mask = dists_phys <= self.d_max
                weights = np.exp(-0.5 * (dists_phys / self.sigma)**2) * mask
                risks = risk_value * np.sum(weights, axis=1)
                total_risks[i:i+batch_size] += risks

        if total_risks.size == 0:
            min_risk, max_risk = 0.0, 1.0
        else:
            min_risk = np.min(total_risks)
            max_risk = np.max(total_risks)

        if max_risk != min_risk:
            norm_risks = (total_risks - min_risk) / (max_risk - min_risk)
        else:
            norm_risks = np.zeros_like(total_risks)

        for idx, coord in enumerate(nonzero_coords):
            i, j, k = coord
            risk_map[i, j, k] = norm_risks[idx]

        return risk_map


if __name__ == "__main__":
    original_data, original_spacing, affine = load_nii_image('Head_label.nii')
    ventricles_segmentation, _, _ = load_nii_image('Ventricles_label.nii')
    vessels_segmentation, _, _ = load_nii_image('Vessels-label.nii')
    m1_segmentation, _, _ = load_nii_image('M1.nii')
    risk_values = [0.45, 0.7, 0.95]
    structures = [ventricles_segmentation, m1_segmentation, vessels_segmentation]
    generator = RiskMapGenerator(
        structures=structures,
        risk_values=risk_values,
        voxel_spacing=original_spacing,
        N=5,
        sigma=5,
        d_max=20.0,
    )
    risk_map = generator.generate_risk_map(original_data, batch_size=50000)
    np.save('RiskMap.npy', risk_map)