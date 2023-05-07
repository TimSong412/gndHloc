from vistool import Vis3D
import numpy as np
from scipy.spatial.transform import Rotation as rot


def main():
    vis3d = Vis3D(
        xyz_pattern=('x', 'y', 'z'),
        out_folder="dbg",
        sequence="vistest",
        # auto_increase=,
        # enable=,
    )
    unit = np.array([1, 0, 0])
    eu1  =  np.array([-1.59638407e-04,  1.70085842e+00,  5.91714433e-02])
    eu2 = np.array([-0.00594446,  1.37288232,  0.02940048])
    eu3 = np.array([-0.00619752,  1.39134822,  0.03110581])
    Rl = rot.from_euler('yxz', [eu1[1], 0, 0], degrees=False)
    Rd = rot.from_euler('yxz', [eu2[1], 0, 0], degrees=False)
    Rn = rot.from_euler('yxz', [eu3[1], 0, 0], degrees=False)
    vec1 = Rl.as_matrix().dot(unit)
    vec2 = Rd.as_matrix().dot(unit)
    vec3 = Rn.as_matrix().dot(unit)
    vis3d.add_lines(np.array([0, 0, 0]), vec1, name="left")
    vis3d.add_lines(np.array([0, 0, 0]), vec2, name="down")
    vis3d.add_lines(np.array([0, 0, 0]), unit, name="unit")
    vis3d.add_lines(np.array([0, 0, 0]), vec3, name="right")

if __name__ == "__main__":
    main()
