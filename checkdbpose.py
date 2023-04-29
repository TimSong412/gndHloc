import numpy as np
from vistool import Vis3D
import trimesh
from scipy.spatial.transform import Rotation as rot
import hloc.utils.read_write_model as rw
from pathlib import Path
from planefit import fitgndplane, gen_squareface
from visloc import viscam_loc, visobj_cam
from Givens import genGivens
from gndloc import read_qimg_intrinsic, intersect_planeXray, visobj_gnd
from viscompare import getimgposemat

def main():
    sfmbase = Path("outputs/MMW_gndsfm/sfm_superpoint+superglue")
    imgs = rw.read_images_binary(sfmbase/"images.bin")
    cams = rw.read_cameras_binary(sfmbase/"cameras.bin")
    pts3d = rw.read_points3D_binary(sfmbase/"points3D.bin")
    vis3d = Vis3D(
        xyz_pattern=('x', 'y', 'z'),
        out_folder="dbg",
        sequence="checkpose",
        # auto_increase=,
        # enable=,
    )
    imgs_kname = {}
    for img in imgs.values():
        imgs_kname[img.name] = img
    demolist = ["1102_MMW_DJI_0076_00161.jpg"]
    st = np.array([0, 0, 0, 1])
    for demo in demolist:
        demoimg = imgs_kname[demo]
        democam = cams[demoimg.camera_id]
        demopose = getimgposemat(demoimg)
        vis3d.add_camera_trajectory(np.array([demopose]), name=demo)
        xyid = np.vstack([demoimg.xys.T, demoimg.point3D_ids]).T
        valid2dpts = xyid[demoimg.point3D_ids>0]
        # [441.5, 455.0, 4804.0]
        pt3d = pts3d[5028]
        ed1 = (demopose@np.array([441.5-democam.params[1], 455.0-democam.params[2], democam.params[0], 1]))[0:3]
        ed2 = pt3d.xyz
        vis3d.add_lines(start_points=(demopose@st)[0:3], end_points = ed1, name="2dpt")
        vis3d.add_lines(start_points=(demopose@st)[0:3], end_points =ed2, name="3Dpt")
    
    verts = []
    rgbs = []
    for pt3d in pts3d.values():
        verts.append(pt3d.xyz)
        rgbs.append(pt3d.rgb)
    vis3d.add_point_cloud(np.array(verts), np.array(rgbs), name="sparse")

if __name__ == "__main__":
    main()