import cv2
import glob
import numpy as np
from vistool import Vis3D
import trimesh
from scipy.spatial.transform import Rotation as rot
import hloc.utils.read_write_model as rw
from pathlib import Path
from planefit import fitgndplane, gen_squareface
from visloc import viscam_loc, visobj_cam
from Givens import genGivens


def read_qimg_intrinsic(qimgfile):
    '''
    return:
    qintrinsics = {imgid: [w, h, foclen, cx, cy, ...]}
    '''
    qintr = {}
    with open(qimgfile) as qf:
        qlines = qf.readlines()
        for img in qlines:
            img = img.split()
            qintr[img[0]] = []
            for param in img[2:]:
                qintr[img[0]].append(float(param))
    return qintr

def intersect_planeXray(planes: np.ndarray, rays: np.ndarray):
    '''
    input:
        planes: n*[a, b, c, d]
        rays: m*[x0, y0, z0, xd, yd, zd]
    output:
        intersects: n*m*[x, y, z]
    '''
    pass
    inters = np.zeros((planes.shape[0], rays.shape[0], 3))
    for pi, p in enumerate(planes):
        for ri, r in enumerate(rays):
            t = (-p[3]-p[0:3]@r[0:3])/(p[0:3]@r[3:6])
            inters[pi, ri] = r[0:3]+t*r[3:6]
    return inters

def visobj_gnd(name, gnd: np.ndarray, objcenters: np.ndarray,  objboxes: np.ndarray=None, vis3d:Vis3D=None):
    if vis3d is not None:        
        for oi, objc in enumerate(objcenters):
            vis3d.add_lines(objc, objc+gnd[0:3], name=name+f"_up{oi}")
            vis3d.add_lines(objc, objc-gnd[0:3], name=name+f"_down{oi}")


def main():
    reconpath = Path("outputs/MMW_sfm/sfm_superpoint+superglue")
    # locpath = Path("outputs/MMW_loc")
    locpath = Path("outputs/MMW_loc_sfm")
    vis3d = Vis3D(
        xyz_pattern=('x', 'y', 'z'),
        out_folder="dbg",
        sequence="gndloc_sfm",
        # auto_increase=,
        # enable=,
    )
    # DB database
    camdata = rw.read_cameras_binary(reconpath/"cameras.bin")
    imgdata = rw.read_images_binary(reconpath/"images.bin")
    pts3data = rw.read_points3D_binary(reconpath/"points3D.bin")

    # query database
    qimgfile = Path("datasets/MMW/qimg_new.txt")
    qimgdetdir = Path("datasets/MMW/query_predict/labels")
    qimgintrinsics = read_qimg_intrinsic(qimgfile=qimgfile)

    [a, b, c, d], inliers, inlierpts = fitgndplane(
        cams=camdata, imgs=imgdata, pts3d=pts3data, maskpth=Path("datasets/MMW/masks_all"))
    
    GVtrans = np.eye(4)
    GVtrans[0:3, 0:3] = genGivens(np.array([a, b, c]))
    GVtrans[2, 3]=d
    # a, b, c, d = GVtrans@np.array([a, b, c, d])
    a, b, c, d = 0, 0, 1, 0
    locposes = viscam_loc(
        locposefile=locpath/"MMW_hloc_superpoint+superglue_netvlad20.txt", vis3d=vis3d, transpose=GVtrans)
    demoimgs = ["1109_MMW_DJI_0001_00019.jpg", "1109_MMW_DJI_0005_00087.jpg", "20211115_AENT7199_00017.jpg"]

    for demo in demoimgs:
        intr = qimgintrinsics[demo]
        detfile = open(qimgdetdir / demo.replace(".jpg", ".txt"), 'r')
        detfile = detfile.readlines()
        detobjs = [line.split() for line in detfile]
        rays = visobj_cam(campose=locposes[demo],
                          w=intr[0],
                          h=intr[1],
                          focl=intr[2],
                          cx=intr[3],
                          cy=intr[4],
                          objlist=detobjs,
                          vis3d=vis3d)
        print("rays= ", rays)
        rays2int = np.array([np.concatenate([rays['startpt'], r['ray']]) for r in rays['objlist']])
        inters = intersect_planeXray(np.array([[a, b, c, d]]), rays2int)
        print("inters= ", inters)
        visobj_gnd(demo, np.array([a, b, c, d]), inters[0], vis3d=vis3d)
        

    densemesh = trimesh.load(
        reconpath/"dense"/"fused.ply")
    meshvert = densemesh.vertices
    homovert = np.vstack([meshvert.T, np.ones(meshvert.shape[0])])
    vis3d.add_point_cloud(GVtrans.dot(homovert).T[..., 0:3],
                          colors=densemesh.colors, name="MMW_sfm")
    facevet = gen_squareface(5, a, b, c, d)
    vis3d.add_mesh(facevet[0:3], name="GNDface1")
    vis3d.add_mesh(facevet[3:6], name="GNDface2")
    vis3d.add_mesh(facevet[[2, 1, 0]], name="GNDface3")
    vis3d.add_mesh(facevet[[5, 4, 3]], name="GNDface4")


if __name__ == "__main__":
    main()
