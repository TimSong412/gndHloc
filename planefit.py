import numpy as np
import trimesh
from vistool import Vis3D
import random
import tqdm
import cv2
from pathlib import Path

import open3d as o3d

from hloc.utils.read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary
import time

def points2plane(pts):
    x0x1 = pts[1] - pts[0]
    x0x2 = pts[2] - pts[0]
    planevec = np.array([x0x1[1]*x0x2[2]-x0x1[2]*x0x2[1], x0x1[2]*x0x2[0]-x0x1[0]*x0x2[2], x0x1[0]*x0x2[1] - x0x1[1]*x0x2[0]])
    return planevec / np.linalg.norm(planevec)


def RANSAC_plane(pts: np.ndarray, max_iter=100, disthresh=1, sample_n=3):
    print("pts= ", pts)
    print("shape= ", pts.shape)
    ptidx = random.sample(range(pts.shape[0]), sample_n)
    chosenpts = pts[ptidx]
    best_inlier_ratio = -1
    res = None
    inlier_r = 0    
    for it in tqdm.trange(max_iter,):
        planevec = points2plane(chosenpts)
        dists = abs((pts-chosenpts[0]).dot(planevec))
        inlier_pts = pts[dists<disthresh]
        inlier_r = sum(dists<disthresh)/pts.shape[0]
        print("inlier= ", inlier_r)
        if inlier_r > best_inlier_ratio:
            best_inlier_ratio = inlier_r
            res = chosenpts
        ptidx = random.sample(range(inlier_pts.shape[0]), sample_n)
        chosenpts = inlier_pts[ptidx]
    print("best_inlier_ratio= ", best_inlier_ratio)
    return res

def selectpts(cameras, images, pts3d, maskpath):
    gndpts = {}
    for img in images.values():
        xyid = np.int64(np.vstack([img.xys.T, img.point3D_ids]).T)
        valid2dpts = xyid[img.point3D_ids>0]
        imgmask = cv2.imread(str(maskpath / img.name), cv2.IMREAD_GRAYSCALE)
        for pt in valid2dpts:
            if imgmask[pt[1], pt[0]] > 0:
                gndpts[pt[2]] = pts3d[pt[2]]
    return gndpts

def gen_squareface(unit, a, b, c, d):
    p1 = -a/c
    p2 = -b/c
    p3 = -d/c
    face = np.array([[unit, unit, p1*unit+p2*unit+p3],
                    [unit, -unit, p1*unit-p2*unit+p3],
                    [-unit, unit, -p1*unit+p2*unit+p3],
                    [-unit, unit, -p1*unit+p2*unit+p3],
                    [unit, -unit, p1*unit-p2*unit+p3],                    
                    [-unit, -unit, -p1*unit-p2*unit+p3]])
    return face

def fitgndplane(cams, imgs, pts3d, maskpth):
    '''
    select ground points
    RANSAC fit plane
    check upside norm
    '''
    t0 = time.time()
    gnd = selectpts(cameras=cams, images=imgs, pts3d=pts3d, maskpath=maskpth)
    t1 = time.time()
    gndarray = np.array([pt.xyz for pt in gnd.values()])
    ptsmean = np.mean(np.array([pt.xyz for pt in pts3d.values()]), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gndarray)
    [a, b, c, d], inliers = pcd.segment_plane(distance_threshold=0.1,
                                            ransac_n=3,
                                            num_iterations=1000)
    inlier_pts = np.asarray(pcd.points)[inliers]
    # outlier_pts = np.asarray(pcd.points)[not inliers]
    if np.array([a, b, c]) @ ptsmean +d < 0:
        a , b, c, d = -a, -b, -c, -d
    t2 = time.time()
    print("select_time= ", t1-t0)
    print("seg_time= ", t2-t1)
    return [a, b, c, d], inliers, inlier_pts

def main():
    basepath = Path("outputs/MMW_sfm/sfm_superpoint+superglue")    
    cameras = read_cameras_binary(basepath/"cameras.bin")
    images = read_images_binary(basepath/"images.bin")
    pts3d = read_points3D_binary(basepath/"points3D.bin")
    gndpts = selectpts(cameras=cameras, images=images, pts3d=pts3d, maskpath=Path("datasets/MMW/masks_all"))
    # print(gndpts)
    gndarray = np.array([pt.xyz for pt in gndpts.values()])
    vis3d = Vis3D(
        xyz_pattern=('x', 'y', 'z'),
        out_folder="dbg",
        sequence="planefit_vis",
    )
    mesh = trimesh.load(
        "outputs/MMW_sfm/sfm_superpoint+superglue/dense/fused.ply")
    vis3d.add_point_cloud(mesh.vertices, colors=mesh.colors, name="MMW_sfm")
    vis3d.add_point_cloud(gndarray, name="gnd")
    # chosenpts = RANSAC_plane(mesh.vertices, disthresh=0.5)
    # print("plane= ", chosenpts)
    # x0x1 = chosenpts[1] - chosenpts[0]
    # x0x2 = chosenpts[2] - chosenpts[0]
    # newx1 = chosenpts[0]+10*x0x1
    # newx2 = chosenpts[0]+10*x0x2
    
    # vis3d.add_mesh(vertices=np.array([chosenpts[0], newx1, newx2]))
    # vis3d.add_mesh(vertices=np.array([newx1, newx2, chosenpts[0]]))

    # print("vetshape= ", mesh.vertices.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gndarray)
    # pcd.normals = o3d.utility.Vector3dVector(mesh.)
    # pcd.colors = o3d.utility.Vector3dVector(np.array(mesh.colors).reshape(-1, 3))
    
    [a, b, c, d], inliers = pcd.segment_plane(distance_threshold=0.1,
                                            ransac_n=3,
                                            num_iterations=1000)
    
    # print("inlier= ", inliers.shape)
    inlier_pts = np.asarray(pcd.points)[inliers]
    vis3d.add_point_cloud(inlier_pts, name="PLANE")
    facevet = gen_squareface(5, a, b, c, d)
    vis3d.add_mesh(facevet[0:3], name="GNDface1")
    vis3d.add_mesh(facevet[3:6], name="GNDface2")

if __name__ == '__main__':
    main()