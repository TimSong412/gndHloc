from vistool import Vis3D
import trimesh
import json
import numpy as np
from pathlib import Path
from planefit import gen_squareface
from visloc import viscam_loc, visobj_cam
import hloc.utils.read_write_model as rw
from gndloc import  intersect_planeXray
from mergedet import LocViser

def main(vis3d: Vis3D=None, d_bias=-0.15, name=None):
    labelid = "00183"
    sfmpath = Path("outputs/MED_sfm_gnd")
    reconpath = Path("outputs/MED_sfm_gnd/sfm_superpoint+superglue")
    labelpath = Path(f"datasets/MED/labels/{labelid}.json")
    
    gndinfo = np.load(sfmpath/"gndinfo.npy", allow_pickle=True).item()
    GVtrans = gndinfo['GVtrans']
    # a, b, c, d = GVtrans@np.array([a, b, c, d])
    a, b, c, d = 0, 0, 1, d_bias

    if name is None:
        name = "vislabel"
    if vis3d is None:
        vis3d = Vis3D(
            xyz_pattern=('x', 'y', 'z'),
            out_folder="dbg",
            sequence=name,
            # auto_increase=,
            # enable=,
        )
        mesh = trimesh.load(
        reconpath/"dense"/"fused.ply")
        colors = mesh.colors
        meshvert = mesh.vertices

        homovert = np.column_stack([meshvert, np.ones(meshvert.shape[0])])
        vis3d.add_point_cloud(GVtrans.dot(homovert.T).T[..., 0:3],
                            colors=colors, name=f"{labelid}_sfm")
        facevet = gen_squareface(10, a, b, c, -0.02)
        vis3d.add_mesh(facevet[0:3], name="GNDface1")
        vis3d.add_mesh(facevet[3:6], name="GNDface2")
        vis3d.add_mesh(facevet[[2, 1, 0]], name="GNDface3")
        vis3d.add_mesh(facevet[[5, 4, 3]], name="GNDface4")



   
    labels = json.load(open(labelpath, 'r'))
    imgs = rw.read_images_binary(reconpath/"images.bin")
    cams = rw.read_cameras_binary(reconpath/"cameras.bin")
    imgkname = {}
    for img in imgs.values():
        imgkname[img.name] = img
    img = imgkname[f"{labelid}.jpg"]
    cam = cams[img.camera_id]
    campose = {img.name: (img.qvec, img.tvec)}
    locposes = viscam_loc(locpose=campose, vis3d=vis3d, transpose=GVtrans)
    locpose = locposes[f"{labelid}.jpg"]   
    rays2int = []
    st = locpose@np.array([0,0, 0, 1])
    w = cam.width
    h = cam.height
    focl = cam.params[0]
    cx = cam.params[1]
    cy = cam.params[2]
    shape_type = None
    for shape in labels["shapes"]:
        if shape["shape_type"] == "line":
            shape_type = "line"
            ed0 = locpose@np.array([float(shape['points'][0][0]) -
                                    cx, float(shape['points'][0][1])-cy, focl, 1])
            ray0 = (ed0-st)[0:3] / np.linalg.norm((ed0-st)[0:3])
            ed1 = locpose@np.array([float(shape['points'][1][0]) -
                                    cx, float(shape['points'][1][1])-cy, focl, 1])
            ray1 = (ed1-st)[0:3] / np.linalg.norm((ed1-st)[0:3])
            rays2int.append([*st[0:3], *ray0])
            rays2int.append([*st[0:3], *ray1])
            # vis3d.add_shapes(st[0:3], ed0[0:3], name=shape['label']+"tail")
            # vis3d.add_shapes(st[0:3], ed1[0:3], name=shape['label']+"head")
        elif shape["shape_type"] == "point":
            shape_type = "point"
            ed0 = locpose@np.array([float(shape['points'][0][0]) -
                                    cx, float(shape['points'][0][1])-cy, focl, 1])
            ray0 = (ed0-st)[0:3] / np.linalg.norm((ed0-st)[0:3])
            rays2int.append([*st[0:3], *ray0])
            

    inters = intersect_planeXray(np.array([[a, b, c, d]]), np.array(rays2int))
    up = np.array([0, 0, 0.1])
    centers = []
    dirs = []
    if shape_type ==  "line":
        for id in range(0, inters[0].__len__(), 2):
            vis3d.add_lines(inters[0][id], inters[0][id+1], name=str(id))
            vis3d.add_lines(inters[0][id], inters[0][id]+up)
            vis3d.add_lines(inters[0][id+1], inters[0][id+1]+up)
            dir = inters[0][id+1] - inters[0][id]
            vis3d.add_boxes(positions=(inters[0][id]+inters[0][id+1])/2, eulers=np.array([0, 0, np.arctan(dir[1]/dir[0])]), extents=np.array([0.2, 0.1, 0.2]), name=f"label{int(id/2)+1}")
            centers.append((inters[0][id]+inters[0][id+1])/2)
            dirs.append(dir)
        
        
        return np.array(centers), np.array(dirs), vis3d
    elif shape_type == "point":           
        
        for id in range(0, inters[0].__len__()):                
            vis3d.add_lines(inters[0][id], inters[0][id]+up)                                
            vis3d.add_boxes(positions=inters[0][id], eulers=np.array([0, 0, 0]), extents=np.array([0.03, 0.03, 0.4]), name=f"label{int(id)+1}")
            centers.append(inters[0][id])                
        
        return np.array(centers), vis3d

    # vis3d.add_boxes_by_dof(positions=np.array([0, 0, 0]), rotations=np.array([0, 0, 0]), scales=np.array([1, 2, 3]), name="box")
    



if __name__ == "__main__":
    centers, vis3d = main(d_bias=-0.05)
    infopath = Path("outputs/MEDped_loc")
    viser = LocViser(infopath, vis3d, cls="ped")
    viser.visdet("00181.jpg")

