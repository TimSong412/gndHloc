import cv2
import glob
import numpy as np
from vistool import Vis3D
import trimesh
from scipy.spatial.transform import Rotation as rot
import hloc.utils.read_write_model as rw
from hloc import match_features_inc
from pathlib import Path
from planefit import fitgndplane, gen_squareface
from visloc import viscam_loc, visobj_cam
from Givens import genGivens
from gndloc import read_qimg_intrinsic, intersect_planeXray, visobj_gnd
from loc_increment import LocDet
import pycolmap


def main():
    datapath = Path("datasets/MMW")
    sfmpath = Path("outputs/MMW_sfm_gnd_lesskpt")
    reconpath = sfmpath / "sfm_superpoint+superglue"
    dense = True
    # locpath = Path("outputs/MMW_loc")
    locpath = Path("outputs/MMW_loc_sfm")
    vis3d = Vis3D(
        xyz_pattern=('x', 'y', 'z'),
        out_folder="dbg",
        sequence="gndloc_inc_lesskpt",
        # auto_increase=,
        # enable=,
    )
    # DB database
    camdata = rw.read_cameras_binary(reconpath/"cameras.bin")
    imgdata = rw.read_images_binary(reconpath/"images.bin")
    pts3data = rw.read_points3D_binary(reconpath/"points3D.bin")
    localizer = LocDet(sfmpath=sfmpath, matcher_conf=match_features_inc.confs['superglue'], num_globalmatch=8)

    # query database
    qimgfile = datapath/"qimg_new.txt"
    qimgdetdir = datapath/ "query_predict/labels"
    qimgintrinsics = read_qimg_intrinsic(qimgfile=qimgfile)

    gndinfo = np.load(sfmpath/"gndinfo.npy", allow_pickle=True).item()

    GVtrans = gndinfo['GVtrans']
    # a, b, c, d = GVtrans@np.array([a, b, c, d])
    a, b, c, d = 0, 0, 1, 0
    # locposes = viscam_loc(
    #     locpose=locpath/"MMW_hloc_superpoint+superglue_netvlad20.txt", vis3d=vis3d, transpose=GVtrans)
    demoimgs = ["1109_MMW_DJI_0001_00019.jpg",
                "1109_MMW_DJI_0005_00087.jpg", "20211115_AENT7199_00017.jpg"]

    for demo in demoimgs:
        intr = qimgintrinsics[demo]
        camerainfo = pycolmap.Camera('SIMPLE_RADIAL', int(intr[0]), int(intr[1]), np.array(intr[2:]))
        image = cv2.imread((datapath/"query"/demo).__str__(), cv2.IMREAD_ANYCOLOR)
        posevec, det = localizer.loc(image=image, imgname=demo, imgintr=camerainfo)
        locposes = viscam_loc(locpose=posevec, vis3d=vis3d, transpose=GVtrans)

        # detfile = open(qimgdetdir / demo.replace(".jpg", ".txt"), 'r')
        # detfile = detfile.readlines()
        # detobjs = [line.split() for line in detfile]
        detobjs = np.column_stack([det.boxes.cls.cpu().numpy(), det.boxes.xywhn.cpu().numpy()])
        rays = visobj_cam(campose=locposes[demo],
                          w=intr[0],
                          h=intr[1],
                          focl=intr[2],
                          cx=intr[3],
                          cy=intr[4],
                          objlist=detobjs,
                          vis3d=vis3d)
        print("rays= ", rays)
        rays2int = np.array(
            [np.concatenate([rays['startpt'], r['ray']]) for r in rays['objlist']])
        inters = intersect_planeXray(np.array([[a, b, c, d]]), rays2int)
        print("inters= ", inters)
        visobj_gnd(demo, np.array([a, b, c, d]), inters[0], vis3d=vis3d)

    if dense:
        mesh = trimesh.load(
            reconpath/"dense"/"fused.ply")
        colors = mesh.colors
        meshvert = mesh.vertices
    else:
        meshvert= np.array([p3d.xyz for p3d in pts3data.values()])
        colors = np.array([p3d.rgb for p3d in pts3data.values()])
        pass
        
    
    homovert = np.column_stack([meshvert, np.ones(meshvert.shape[0])])
    vis3d.add_point_cloud(GVtrans.dot(homovert.T).T[..., 0:3],
                          colors=colors, name="MMW_sfm")
    facevet = gen_squareface(5, a, b, c, d)
    vis3d.add_mesh(facevet[0:3], name="GNDface1")
    vis3d.add_mesh(facevet[3:6], name="GNDface2")
    vis3d.add_mesh(facevet[[2, 1, 0]], name="GNDface3")
    vis3d.add_mesh(facevet[[5, 4, 3]], name="GNDface4")


if __name__ == "__main__":
    main()
