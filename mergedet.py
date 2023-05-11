import numpy as np
import vislabel
from vistool import Vis3D
from pathlib import Path
import os
import glob
import cv2
import lpips
import torch
from KMmatch import KM_matcher


def dist(p1, p2):
    return np.linalg.norm((p1[0:2]-p2[0:2]))


def angle(v1, v2):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    return np.arccos(v1@v2) * 180 / np.pi


def evaluate(predlocs, preddirs, labellocs, labeldirs, distthresh=0.1, result=None):
    if result is None:
        result = {}
    print("VALUATE RESULT:")
    cnt_suc = 0
    cnt_fal = 0
    dist_bias = 0
    angel_bias = 0
    for id, loc in enumerate(predlocs):
        # MED region:
        if loc[0] > 0:
            continue

        bestid = -1
        bestdist = 1e10
        for lid, lloc in enumerate(labellocs):
            if dist(loc, lloc) < bestdist and dist(loc, lloc) < distthresh:
                bestid = lid
                bestdist = dist(loc, lloc)
        if bestid == -1:
            cnt_fal += 1
            print("Loc Fail, x, y = ", loc[0:2])
            result[id] = [0, *loc[0:2], *preddirs[id][0:2]]
        else:
            cnt_suc += 1
            min_angle_bias = min(angle(preddirs[id][0:2], labeldirs[bestid][0:2]), angle(
                preddirs[id][0:2], -labeldirs[bestid][0:2]))
            angel_bias += min_angle_bias
            dist_bias += bestdist
            print("loc SUCCESS, xy = ", loc[0:2], " label= ", labellocs[bestid][0:2],
                  f"distance = {bestdist: .3f}", f" min_dir_bias = {min_angle_bias: .2f} degree")
            # [status, loc_x, loc_y, rot_x, rot_y, label_x, label_y, label_rot_x, label_rot_y, distance, angle]
            result[id] = [1, *loc[0:2], *preddirs[id][0:2],
                          *labellocs[bestid][0:2], *labeldirs[bestid][0:2], bestdist, min_angle_bias]
    print("SUCCESS_CNT= ", cnt_suc)
    print("FAIL_CNT= ", cnt_fal)
    print("SUCC_RATE= ", cnt_suc / (cnt_suc + cnt_fal))
    if cnt_suc > 0:
        print("MEAN_DIST_ERR= ", dist_bias / cnt_suc)
        print("MEAN_ANGLE_ERR= ", angel_bias/cnt_suc)
    print("--------------------------------------------------------")
    print("\n")
    return result


class LocViser():
    def __init__(self, infopath: Path, vis3d: Vis3D = None, d_bias=-0.15) -> None:
        self.infopath = infopath
        self.locdir = infopath / "locinfo"
        self.cropdir = infopath / "crops"
        self.vis3d = vis3d
        self.z = -d_bias

    def visboxes(self, centers, dirs, name: str):
        for id, loc in enumerate(centers):
            if len(loc) == 2:
                loc = np.array([*loc, self.z])
            self.vis3d.add_boxes(positions=loc, eulers=np.array([0, 0, np.arctan(
                dirs[id][1]/dirs[id][0])]), extents=np.array([0.2, 0.1, 0.4]), name=f"{name}_{id}")

    def visdet(self, name, vis=True):
        locinfo = np.load(
            self.locdir / f"{name.split('.')[0]}.npy", allow_pickle=True).item()
        if vis:
            self.visboxes(locinfo['locs'], locinfo['dirs'], name)
        return locinfo


def eval():
    # car length: 5m, width: 1.8m MED average: 0.37, half width = 0.07, half length = 0.185
    labellocs, labeldirs, vis3d = vislabel.main(d_bias=-0.15)
    infopath = Path("outputs/MED_loc")
    mgr = LocViser(infopath, vis3d)
    demoimgs = [
        "00021.jpg",  # d = -0.1
        "00087.jpg",  # d = -0.15
        "00097.jpg",
        "00102.jpg"]
    demo = "00021.jpg"
    info1 = mgr.visdet(demo)

    thresh = 0.185

    resultpath = infopath / "results" / f"{demo}_result_{thresh:.3f}.npy"
    if os.path.exists(resultpath):
        result = np.load(resultpath, allow_pickle=True).item()
    else:
        result = {}
    if demo not in result:
        result[demo] = {}

    res = result[demo]

    res = evaluate(info1['locs'], info1['dirs'],
                  labellocs, labeldirs, distthresh=thresh, result=res)
    result[demo] = res
    # np.save(resultpath, result)


class Obj():
    def __init__(self, id, imgs: list = None, w_img=None, centers: list = None, w_center=None, dirs: list = None, w_dir=None, camposes: list = None, imgsize=128) -> None:
        self.id = id
        self.camposes = camposes
        self.imgs = imgs
        self.centers = centers
        self.dirs = dirs
        self.w_img = w_img
        self.w_center = w_center
        self.w_dir = w_dir
        self.imgsz = imgsize

    def read_imgs(self) -> list:
        imgs = []
        for imgfile in self.imgs:
            im = cv2.imread(imgfile.__str__(), cv2.IMREAD_ANYCOLOR)
            im = cv2.resize(im, (self.imgsz, self.imgsz)).astype(np.float32)
            im = 2*(im-im.min())/(im.max()-im.min())-1.0
            imgs.append(im.transpose(2, 0, 1))
        return imgs

    def get_center(self, top=False):
        if top:
            return self.centers[np.argmax(self.w_center)]
        else:
            if len(self.w_center) == 1 and self.w_center[0] == 0:
                return self.centers[0]
            meancenter = np.sum((np.array(
                self.w_center)*np.array(self.centers).T).T, axis=0) / np.sum(self.w_center)
            return meancenter

    def get_dir(self, top=True):
        if top:
            return self.dirs[np.argmax(self.w_dir)]
        else:
            # meancenter = (np.array(self.w_center)*np.array(self.centers).T).T
            # return np.sum(meancenter, axis=0)
            raise NotImplementedError()


class Observe():
    def __init__(self, name, croppath: Path, locinfo, objlist: dict = None, height_bias=0.15, max_dist=5.92) -> None:
        self.name = name
        self.croppath = croppath
        self.d_bias = height_bias
        self.max_dist = 5.92
        if isinstance(locinfo, Path):
            self.locinfo = [np.load(locinfo, allow_pickle=True).item()]
        else:
            self.locinfo = locinfo

        if objlist is not None:
            self.construct_raw(objlist)
        else:
            self.objs = {}
    def obs_dist(self, center):
        return np.linalg.norm([(self.locinfo[0]['campose'][2, 3]-self.d_bias), np.linalg.norm((
                self.locinfo[0]['campose'][0:2, 3]-center[0:2]))])
    
    def construct_raw(self, objlist: dict):
        self.objs = {}
        for k in objlist.keys():
            imgs = [self.croppath / f"{self.name}_{k}.jpg"]
            
            if self.obs_dist(objlist[k][1:3]) > self.max_dist:
                w_dir = 10
                w_cen = 1e-10
            else:
                w_dir = w_cen = observe_pitch = np.arctan((self.locinfo[0]['campose'][2, 3]-self.d_bias)/np.linalg.norm(
                self.locinfo[0]['campose'][0:2, 3]-objlist[k][1:3]))
            
            self.objs[f"{self.name}_{k}"] = Obj(f"{self.name}_{k}", imgs=imgs, w_img=[1], centers=[
                objlist[k][1:3]], w_center=[w_cen], dirs=[objlist[k][3:5]], w_dir=[w_dir], camposes=[self.locinfo[0]['campose']])

    def getobjinfo(self):
        locs = []
        dirs = []
        for obj in self.objs.values():
            locs.append(obj.get_center())
            dirs.append(obj.get_dir())
        return {'name': self.name, 'locs': np.array(locs), 'dirs': np.array(dirs)}


class Merger():
    def __init__(self, dist_thresh=0.37, match_loc_bias=0.99) -> None:
        self.loc_bias = match_loc_bias
        self.dist_thresh = dist_thresh
        self.loss = lpips.LPIPS(net='alex')

    def merge(self, ob0: Observe, ob1: Observe) -> Observe:
        matches = self.matchobs(ob0, ob1)
        if matches is None:
            return None
        newobs = Observe(name=ob0.name+"&"+ob1.name,
                         croppath=ob0.croppath, locinfo=[*ob0.locinfo, *ob1.locinfo])
        ob0matched = matches[..., 0]
        ob1matched = matches[..., 1]
        id_cnt = 0
        for id0 in ob0.objs.keys():
            if id0 not in ob0matched:
                newobs.objs[id0] = ob0.objs[id0]
                newobs.objs[id0].id = ob0.name+"&"+ob1.name+f"_{id_cnt}"
                id_cnt += 1
        for id1 in ob1.objs.keys():
            if id1 not in ob1matched:
                newobs.objs[id1] = ob1.objs[id1]
                newobs.objs[id1].id = ob0.name+"&"+ob1.name+f"_{id_cnt}"
                id_cnt += 1
        # Fusion
        for m in matches:
            newid = m[0]+"&"+m[1]+f"_{id_cnt}"
            id_cnt += 1
            obj0 = ob0.objs[m[0]]
            obj1 = ob1.objs[m[1]]
            # new_w_center = np.sum(obj0.w_center)+np.sum(obj1.w_center)
            # newcenter = (obj0.get_center()*np.sum(obj0.w_center) +
            #              obj1.get_center()*np.sum(obj1.w_center)) / new_w_center
            newcenter = [*obj0.centers, *obj1.centers]
            new_w_center = [*obj0.w_center, *obj1.w_center]
            if obj0.w_dir < obj1.w_dir:
                newdir = obj0.dirs
                new_w_dir = obj0.w_dir
            else:
                newdir = obj1.dirs
                new_w_dir = obj1.w_dir
            newpose = [*obj0.camposes, *obj1.camposes]
            newimgs = [*obj0.imgs, *obj1.imgs]
            new_w_img = [*obj0.w_img, *obj1.w_img]
            newobs.objs[newid] = Obj(newid, newimgs, new_w_img, newcenter, 
                                     new_w_center, newdir, new_w_dir, newpose, 128)
        return newobs

    def matchobs(self, ob0: Observe, ob1: Observe) -> list:
        idlist_0 = []
        idlist_1 = []
        # filter: has at least one candidate
        for i0 in ob0.objs.keys():
            valid = False
            for i1 in ob1.objs.keys():
                if self.objdist(ob0.objs[i0], ob1.objs[i1]) < self.dist_thresh:
                    valid = True
                    break
            if valid:
                idlist_0.append(i0)
        for i1 in ob1.objs.keys():
            valid = False
            for i0 in idlist_0:
                if self.objdist(ob0.objs[i0], ob1.objs[i1]) < self.dist_thresh:
                    valid = True
                    break
            if valid:
                idlist_1.append(i1)
        if idlist_0.__len__() == 0 or idlist_1.__len__() == 0:
            return None

        # len(0) <= len(1)
        # if idlist_0.__len__() > idlist_1.__len__():
        #     ob0, ob1 = ob1, ob0
        #     idlist_0, idlist_1 = idlist_1, idlist_0
        km_edgs = np.zeros((idlist_0.__len__(), idlist_1.__len__()))
        for i, id0 in enumerate(idlist_0):
            for j, id1 in enumerate(idlist_1):
                dist01 = self.objdist(obj0=ob0.objs[id0], obj1=ob1.objs[id1])
                if dist01 > self.dist_thresh:
                    km_edgs[i, j] = -10
                else:
                    km_edgs[i, j] = 1.0/(self.loc_bias*self.objdist(obj0=ob0.objs[id0], obj1=ob1.objs[id1])+(
                        1-self.loc_bias)*self.objsimilar(obj0=ob0.objs[id0], obj1=ob1.objs[id1]))
        km = KM_matcher(km_edgs)
        
        km.Kuh_Munkras()
        matchlist, _ = km.calculateSum()
        finalmatch = []
        # KM match: is complete match
        # filter
        for m in matchlist:
            if self.objdist(obj0=ob0.objs[idlist_0[m[0]]], obj1=ob1.objs[idlist_1[m[1]]]) < self.dist_thresh:
                finalmatch.append([idlist_0[m[0]], idlist_1[m[1]]])
        # finalmatch = [[idlist_0[m[0]], idlist_1[m[1]]] for m in matchlist]
        print("MERGE_MATCH= ", finalmatch)
        return np.array(finalmatch)

    def objdist(self, obj0: Obj, obj1: Obj):
        dist_mat = np.zeros((obj0.w_center.__len__(), obj1.w_center.__len__()))
        for i, cent0 in enumerate(obj0.centers):
            for j, cent1 in enumerate(obj1.centers):
                dist_mat[i, j] = (obj0.w_center[i]*obj1.w_center[j]) * dist(np.array(cent0), np.array(cent1))
        return dist_mat.sum() / (np.sum(obj0.w_center)*np.sum(obj1.w_center))

    def objsimilar(self, obj0: Obj, obj1: Obj):
        similar_mat = np.zeros((obj0.w_img.__len__(), obj1.w_img.__len__()))
        for i, im0 in enumerate(obj0.read_imgs()):
            for j, im1 in enumerate(obj1.read_imgs()):
                similar_mat[i, j] = (obj0.w_img[i]*obj1.w_img[j]) * \
                    self.loss(torch.from_numpy(
                        im0[None]), torch.from_numpy(im1[None]))
        return similar_mat.sum() / (np.sum(obj0.w_img)*np.sum(obj1.w_img))


if __name__ == "__main__":
    # eval()
    car_length = 0.37
    car_width = 0.14
    labellocs, labeldirs, vis3d = vislabel.main(d_bias=-0.05, name="vismerge")
    infopath = Path("outputs/MED_loc")
    viser = LocViser(infopath, vis3d)
    # demo = "00102.jpg"
    # info0 = viser.visdet(demo)
    # demo = "00087.jpg"
    # info1 = viser.visdet(demo)
    res = glob.glob("outputs/MED_loc/results/*0.070.npy")

    # demoimgs = [
    #     "00021.jpg",  # d = -0.1
    #     "00087.jpg",  # d = -0.15
    #     "00097.jpg",
    #     "00102.jpg"]
    demoimgs = [   
        "00120.jpg",
        "DJI_0195_q1.jpg",
        "DJI_0192_q2.jpg"] # d = -0.05
    
    locinfopath = Path("outputs/MED_loc/locinfo")
    croppath = Path("outputs/MED_loc/crops")

    val_thresh = car_width/2
    id0 = "00120"
    info0 = viser.visdet(f"{id0}.jpg", vis=False)
    res0 = evaluate(info0['locs'], info0['dirs'],
                   labellocs, labeldirs, distthresh=val_thresh)
    
    ob0 = Observe(name=f"{id0}.jpg", croppath=croppath,
                  locinfo=locinfopath/f"{id0}.npy", objlist=res0)
    
    id1 = "DJI_0195_q1"
    info1 = viser.visdet(f"{id1}.jpg", vis=False)
    res1 = evaluate(info1['locs'], info1['dirs'],
                   labellocs, labeldirs, distthresh=val_thresh)
    ob1 = Observe(name=f"{id1}.jpg", croppath=croppath,
                  locinfo=locinfopath/f"{id1}.npy", objlist=res1)
    
    id2 = "DJI_0192_q2"
    info2 = viser.visdet(f"{id2}.jpg")
    res2 = evaluate(info2['locs'], info2['dirs'],
                   labellocs, labeldirs, distthresh=val_thresh)
    ob2 = Observe(name=f"{id2}.jpg", croppath=croppath,
                  locinfo=locinfopath/f"{id2}.npy", objlist=res2)
    
    # ob3 = Observe(name="00021.jpg", croppath=croppath,
    #               locinfo=locinfopath/"00021.npy", objlist=results["00021.jpg"]["00021.jpg"])
    
   
    mg = Merger(dist_thresh=car_length)
    newobs = mg.merge(ob0, ob1)

    

    # newobs = mg.merge(ob2, newobs)

    mergeinfo = newobs.getobjinfo()
    
    viser.visboxes(centers=mergeinfo['locs'],
                   dirs=mergeinfo['dirs'], name=mergeinfo['name'])
    resmerge = evaluate(mergeinfo['locs'], mergeinfo['dirs'],
                labellocs, labeldirs, distthresh=val_thresh)


    newobs = mg.merge(newobs, ob2)
    mergeinfo = newobs.getobjinfo()
    
    viser.visboxes(centers=mergeinfo['locs'],
                   dirs=mergeinfo['dirs'], name=mergeinfo['name'])
    resmerge = evaluate(mergeinfo['locs'], mergeinfo['dirs'],
                labellocs, labeldirs, distthresh=val_thresh)

    # newobs = mg.merge(ob3, newobs)
    
    # info3 = viser.visdet("00021.jpg")    
    
    
    # # res0 = evaluate(info0['locs'], info0['dirs'],
    # #                labellocs, labeldirs, distthresh=val_thresh)
    # # res1 = evaluate(info1['locs'], info1['dirs'],
    # #                labellocs, labeldirs, distthresh=val_thresh)


    
    # res3 = evaluate(info3['locs'], info3['dirs'],
    #                labellocs, labeldirs, distthresh=val_thresh)
    
    

    # mergeinfo = newobs.getobjinfo()                   
    # resmerge = evaluate(mergeinfo['locs'], mergeinfo['dirs'],
    #                    labellocs, labeldirs, distthresh=val_thresh)
    # viser.visboxes(centers=mergeinfo['locs'],
    #             dirs=mergeinfo['dirs'], name=mergeinfo['name'])
    
    