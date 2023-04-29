import hloc.utils.read_write_model as rw
from pathlib import Path
from scipy.spatial.transform import Rotation as rot
import numpy as np
import os
from vistool import Vis3D
import trimesh

def getimgposemat(imgdata):
    camrot = rot.from_quat([float(imgdata.qvec[1]), float(
    imgdata.qvec[2]), float(imgdata.qvec[3]), float(imgdata.qvec[0])])
    camtrans = np.array(
        [float(imgdata.tvec[0]), float(imgdata.tvec[1]), float(imgdata.tvec[2])])
    camrot = camrot.as_matrix()
    camrot = camrot.T
    camtrans = -camrot@camtrans
    cammat = np.eye(4)
    cammat[0:3, 0:3] = camrot
    cammat[0:3, 3] = camtrans
    return cammat

def getlocposes(posefile):
    posefile = open(posefile, 'r')
    poses = posefile.readlines()
    camposes = {}
    for pose in poses:
        name = pose.split()[0]
        pose = pose.strip('\n').split()[1:]
        # print(pose)
        camrot = rot.from_quat([float(pose[1]), float(
            pose[2]), float(pose[3]), float(pose[0])])
        camtrans = np.array([float(pose[4]), float(pose[5]), float(pose[6])])
        cammat = np.eye(4)
        camrot = camrot.as_matrix()
        camrot = camrot.T
        camtrans = -camrot@camtrans
        # camtrans = -camrot@camtrans
        # camrot = camrot.as_matrix().T
        # camrot = camrot[..., [2, 0, 1]] * np.array([1, 1, -1])
        cammat[0:3, 0:3] = camrot
        cammat[0:3, 3] = camtrans
        camposes[name] = cammat
    return camposes

def main():
    all_pth = Path("outputs/MMW_all/sfm_superpoint+superglue")
    sfm_pth = Path("outputs/MMW_sfm/sfm_superpoint+superglue")
    query_pth = Path("datasets/MMW/query")
    loc_pth = Path("outputs/MMW_loc_sfm/sfm_superpoint+superglue")
    img_all = rw.read_images_binary(all_pth/"images.bin")
    img_sfm = rw.read_images_binary(sfm_pth/"images.bin")
    img_all_kname = {}
    for img in img_all.values():
        img_all_kname[img.name] = img
    # print(img_all)
    vis3d = Vis3D(
        xyz_pattern=('x', 'y', 'z'),
        out_folder="dbg",
        sequence="vis_compare",
        # auto_increase=,
        # enable=,
    )
    trans_all = []
    poses_all = []
    for img in img_sfm.values():
        if img_all_kname.__contains__(img.name):
            sfm_campose = getimgposemat(img)
            all_campose = getimgposemat(img_all_kname[img.name])
            # P_sfm = trans @ P_all
            trans = sfm_campose @ np.linalg.inv(all_campose)
            print(trans)
            trans_all.append(trans)
            vis3d.add_camera_trajectory(np.array([sfm_campose]), name=img.name+"sfm")
            # vis3d.add_camera_trajectory(np.array([trans@all_campose]), name=img.name+"all")
            poses_all.append([img.name+'all', all_campose])

    trans_all = np.array(trans_all)
    trans_avg = np.average(trans_all, axis=0)
    print("AVG= ", trans_avg)
    for ap in poses_all:
         vis3d.add_camera_trajectory(np.array([trans_avg@ap[1]]), name=ap[0])
    # TODO: PCA get average trans

    query_poses = getlocposes("outputs/MMW_loc_sfm/MMW_hloc_superpoint+superglue_netvlad20.txt")
    poses_all = {}
    for name, locpose in query_poses.items():
        if not img_all_kname.__contains__(name):
            continue
        allpose = getimgposemat(img_all_kname[name])
        allpose_sfm = trans_avg @ allpose
        print("locpose= ", locpose)
        print("allpose_sfm= ", allpose_sfm)
        vis3d.add_camera_trajectory(poses=np.array([locpose]), name=name+"loc")
        vis3d.add_camera_trajectory(poses=np.array([allpose_sfm]), name=name+"all")
        poses_all[name] = allpose_sfm
    np.save("posesall.npy", poses_all)
    densemesh = trimesh.load("outputs/MMW_sfm/sfm_superpoint+superglue/dense/fused.ply")
    vis3d.add_point_cloud(densemesh.vertices,
                          colors=densemesh.colors, name="MMW_sfm")

if __name__ == "__main__":
    main()