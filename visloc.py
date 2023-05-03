from vistool import Vis3D
import trimesh
from scipy.spatial.transform import Rotation as rot
import numpy as np


def viscam_loc(locpose, vis3d: Vis3D = None, transpose:np.ndarray=None):
    '''
    visualize localized query cameras
    params:
        locpose: filename or dict {name: (qvec, tvec), ...}
    '''
    if transpose is None:
        transpose = np.ones((4))
    if isinstance(locpose, dict):
        poses = [[k, *(list(locpose[k][0])+list(locpose[k][1]))] for k in locpose.keys()]
    else:
        posefile = open(locpose, 'r')
        poses = posefile.readlines()
        poses = [pline.strip('\n').split() for pline in poses]
    camposes = {}
    # vis3d.add_camera_trajectory(poses=np.array([np.eye(4)]), name="base")
    for pose in poses:
        # camera center = -R.T * transpose
        name = pose[0]
        pose = pose[1:]
        # print(pose)
        camrot = rot.from_quat([float(pose[1]), float(
            pose[2]), float(pose[3]), float(pose[0])])
        camtrans = np.array([float(pose[4]), float(pose[5]), float(pose[6])])
        cammat = np.eye(4)
        camrot = camrot.as_matrix()
        camrot = camrot.T
        camtrans = -camrot@camtrans

        cammat[0:3, 0:3] = camrot
        cammat[0:3, 3] = camtrans
        camposes[name] = transpose @ cammat
        # print("0= ", cammat)
        
        if vis3d is not None:
            vis3d.add_camera_trajectory(poses=np.array([transpose@cammat]), name=name+"loc")
    return camposes
        # if "20211115_AENT7199_00017" in name:
        #     demopose = cammat
    # print("pose= ", demopose)
    # with open("outputs/MMW_loc/1109_MMW_DJI_0005_00087.txt", 'r') as labelf:
    #     labels = labelf.readlines()
    #     print("labels=", labels)
    #     for label in labels:
    #         print("label= ", label)
    #         label = label.split()
    #         st = demopose@np.array([0, 0, 0, 1])
    #         ed = demopose@np.array([float(label[1])*1920 -
    #                                960, float(label[2])*1080-540, 2304.0, 1])
    #         vis3d.add_lines(start_points=st[0:3], end_points=ed[0:3])

    # rot1 = rot.from_quat([0.022269794529918764, 0.6955744428753393,
    #                      0.6810838547498564, -0.2276071911634924])
    # cam1 = np.eye(4)
    # # rot.from_quat()
    # cam1[0:3, 0:3] = rot1.as_matrix()
    # cam1[0:3, 3] = -np.array([-0.0032048618548638727, -
    #                          0.6055089002120254, 5.555026883911614])
    # print(cam1)

def visobj_cam(campose:np.ndarray, w, h, focl, cx, cy, objlist:list, vis3d: Vis3D=None):
    '''
    objlist: n*(id, cx, cy, w, h), in ratio
    return:
    dict{startpt: xyz, objlist:[{categoryid: int, ray: xyz}]}
    '''
    # with open(labelpth, 'r') as labelf:
    #     labels = labelf.readlines()
    #     print("labels=", labels)    
    st = campose@np.array([0, 0, 0, 1])
    objrays = {"startpt": st[0:3], "objlist": []}
    for objlabel in objlist:
        print("label= ", objlabel)
        # label = label.split()            
        ed = campose@np.array([float(objlabel[1])*w -
                                cx, float(objlabel[2])*h-cy, focl, 1])
        ray = (ed-st)[0:3] / np.linalg.norm((ed-st)[0:3])
        objrays["objlist"].append({"categoryid": int(objlabel[0]), "ray":ray})
        if vis3d is not None:
            vis3d.add_lines(start_points=st[0:3], end_points=ed[0:3])
    return objrays

def getsparsepts(points3d):
    '''
    params: points3d
    return: pts: np.ndarray, rgbs: np.ndarray
    '''
    pts = []
    rgbs = []
    for pt in points3d.values():
        pts.append(pt.xyz)
        rgbs.append(pt.rgb)
    return np.array(pts), np.array(rgbs)


def visdetect_dataimg(name: str, vis3d: Vis3D, imgs: dict, cams: dict):
    for k, v in imgs.items():
        if name in v.name:
            camrot = rot.from_quat([float(v.qvec[1]), float(
                v.qvec[2]), float(v.qvec[3]), float(v.qvec[0])])
            camtrans = np.array(
                [float(v.tvec[0]), float(v.tvec[1]), float(v.tvec[2])])
            camrot = camrot.as_matrix()
            camrot = camrot.T
            camtrans = -camrot@camtrans
            cammat = np.eye(4)
            cammat[0:3, 0:3] = camrot
            cammat[0:3, 3] = camtrans
            imgcam = cams[v.camera_id]
            flen = imgcam.params[0]
            width = imgcam.width
            height = imgcam.height
            cx = imgcam.params[1]
            cy = imgcam.params[2]
            with open(f"datasets/MMW/images_predict/labels/{name}.txt", 'r') as labelf:
                labels = labelf.readlines()
                for label in labels:
                    print("label= ", label)
                    label = label.split()
                    st = cammat@np.array([0, 0, 0, 1])
                    ed = cammat@np.array([float(label[1])*width -
                                          cx, float(label[2])*height-cy, flen, 1])
                    vis3d.add_lines(start_points=st[0:3], end_points=ed[0:3])


if __name__ == "__main__":

    # exit()
    import hloc.utils.read_write_model as rw
    vis3d = Vis3D(
        xyz_pattern=('x', 'y', 'z'),
        out_folder="dbg",
        sequence="locdetect_vis",
        # auto_increase=,
        # enable=,
    )
    _ = viscam_loc(
        locposefile="outputs/MMW_loc/MMW_hloc_superpoint+superglue_netvlad20.txt", vis3d=vis3d)
    imgs = rw.read_images_binary(
        "outputs/MMW_all/sfm_superpoint+superglue/images.bin")
    oriimgs = rw.read_images_binary(
        "outputs/MMW_sfm/sfm_superpoint+superglue/images.bin")
    cams = rw.read_cameras_binary(
        "outputs/MMW_all/sfm_superpoint+superglue/cameras.bin")
    mesh = trimesh.load(
        "outputs/MMW_sfm/sfm_superpoint+superglue/dense/fused.ply")
    vis3d.add_point_cloud(mesh.vertices, colors=mesh.colors, name="MMW_sfm")
    a = imgs[1]
    b = oriimgs[1]

    newrot = rot.from_quat([float(a.qvec[1]), float(
        a.qvec[2]), float(a.qvec[3]), float(a.qvec[0])])
    newtrans = np.array(
        [float(a.tvec[0]), float(a.tvec[1]), float(a.tvec[2])])
    newrot = newrot.as_matrix()
    newrot = newrot.T
    newtrans = -newrot@newtrans
    newmat = np.eye(4)
    newmat[0:3, 0:3] = newrot
    newmat[0:3, 3] = newtrans

    orirot = rot.from_quat([float(b.qvec[1]), float(
        b.qvec[2]), float(b.qvec[3]), float(b.qvec[0])])
    oritrans = np.array(
        [float(b.tvec[0]), float(b.tvec[1]), float(b.tvec[2])])
    orirot = orirot.as_matrix()
    orirot = orirot.T
    oritrans = -orirot@oritrans
    orimat = np.eye(4)
    orimat[0:3, 0:3] = orirot
    orimat[0:3, 3] = oritrans

    transpose = orimat@np.linalg.inv(newmat)

    for k, v in imgs.items():
        print("v= ", v)
        camrot = rot.from_quat([float(v.qvec[1]), float(
            v.qvec[2]), float(v.qvec[3]), float(v.qvec[0])])
        camtrans = np.array(
            [float(v.tvec[0]), float(v.tvec[1]), float(v.tvec[2])])
        camrot = camrot.as_matrix()
        camrot = camrot.T
        camtrans = -camrot@camtrans
        cammat = np.eye(4)
        cammat[0:3, 0:3] = camrot
        cammat[0:3, 3] = camtrans
        vis3d.add_camera_trajectory(
            poses=np.array([transpose@cammat]), name=v.name)
    # visdetect_dataimg("1102_MMW_DJI_0076_00377", vis3d, imgs, cams)
    # visdetect_dataimg("1102_MMW_DJI_0076_00217", vis3d, imgs, cams)
    # visdetect_dataimg("1102_MMW_DJI_0076_00241", vis3d, imgs, cams)
