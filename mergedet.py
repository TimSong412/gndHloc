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


def valuate(predlocs, preddirs, labellocs, labeldirs, distthresh=0.1, result=None):
    if result is None:
        result = {}
    print("VALUATE RESULT:")
    cnt_suc = 0
    cnt_fal = 0
    dist_bias = 0
    angel_bias = 0
    for id, loc in enumerate(predlocs):
        # MED region:
        if loc[1] < -8:
            continue
        if loc[1] > -1:
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
    print("MEAN_DIST_BIAS= ", dist_bias / cnt_suc)
    print("MEAN_ANGLE_BIAS= ", angel_bias/cnt_suc)
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

    def visdet(self, name):
        locinfo = np.load(
            self.locdir / f"{name.split('.')[0]}.npy", allow_pickle=True).item()
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

    res = valuate(info1['locs'], info1['dirs'],
                  labellocs, labeldirs, distthresh=thresh, result=res)
    result[demo] = res
    # np.save(resultpath, result)


# class KMmatcher():
#     def __init__(self, id_list0:list, id_list1:list, kmedges:np.ndarray, dsc=0.01) -> None:
#         self.idlist0 = id_list0
#         self.idlist1 = id_list1
#         self.edgs = kmedges
#         self.heading0 = np.max(kmedges, axis=1)
#         self.heading1 = np.zeros(id_list1.__len__())
#         self.matches = np.zeros((id_list0.__len__(), id_list1.__len__()))
#         self.dsc= dsc

#     def match(self)->list:
#         # waitlist: 1 == not matched, 0 == matched or failed
#         self.wl = np.ones(self.idlist0.__len__())
#         while self.wl.sum() > 0:
#             for i in range(self.wl.__len__()):
#                 if self.wl[i] > 0:
#                     self.search(i)

#         result = np.where(self.matches==1)
#         finalmatch = []
#         for id in range(result[0].__len__()):
#             finalmatch.append([self.idlist0[result[0][id]], self.idlist1[result[1][id]]])
#         return finalmatch

#     def search(self, id):
#         # find a match for list0 [id]
#         candidates = np.argsort(-self.edgs[id])
#         for cid in candidates:
#             if self.edgs[id, cid] >= (self.heading0[id]+self.heading1[cid]):
#                 if np.sum(self.matches[..., cid]) == 0:
#                     self.matches[id, cid] = 1
#                     self.wl[id] = 0
#                     return
#                 elif np.sum(self.matches[..., cid]) > 0:
#                     challenger = np.where(self.matches[id, ...]==1)[0]
#             else:
#                 continue
#         self.wl[id] = 0
#         return


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
            meancenter = (np.array(
                self.w_center)*np.array(self.centers).T).T / np.linalg.norm(self.w_center)
            return np.sum(meancenter, axis=0)

    def get_dir(self, top=True):
        if top:
            return self.dirs[np.argmax(self.w_dir)]
        else:
            # meancenter = (np.array(self.w_center)*np.array(self.centers).T).T
            # return np.sum(meancenter, axis=0)
            raise NotImplementedError()


class Observe():
    def __init__(self, name, croppath: Path, locinfo, objlist: dict = None, height_bias=0.15) -> None:
        self.name = name
        self.croppath = croppath
        self.d_bias = height_bias
        if isinstance(locinfo, Path):
            self.locinfo = [np.load(locinfo, allow_pickle=True).item()]
        else:
            self.locinfo = locinfo

        if objlist is not None:
            self.construct_raw(objlist)
        else:
            self.objs = {}

    def construct_raw(self, objlist: dict):
        self.objs = {}
        for k in objlist.keys():
            imgs = [self.croppath / f"{self.name}_{k}.jpg"]
            observe_pitch = np.arctan((self.locinfo[0]['campose'][2, 3]-self.d_bias)/np.linalg.norm(
                self.locinfo[0]['campose'][0:2, 3]-objlist[k][1:3]))
            self.objs[f"{self.name}_{k}"] = Obj(f"{self.name}_{k}", imgs=imgs, w_img=[1], centers=[
                objlist[k][1:3]], w_center=[observe_pitch], dirs=[objlist[k][3:5]], w_dir=[observe_pitch], camposes=[self.locinfo[0]['campose']])

    def getobjinfo(self):
        locs = []
        dirs = []
        for obj in self.objs.values():
            locs.append(obj.centers[0])
            dirs.append(obj.dirs[0])
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
            new_w_center = np.sum(obj0.w_center)+np.sum(obj1.w_center)
            newcenter = (obj0.get_center()*np.sum(obj0.w_center) +
                         obj1.get_center()*np.sum(obj1.w_center)) / new_w_center
            if obj0.w_dir < obj1.w_dir:
                newdir = obj0.dirs
                new_w_dir = obj0.w_dir
            else:
                newdir = obj1.dirs
                new_w_dir = obj1.w_dir
            newpose = [*obj0.camposes, *obj1.camposes]
            newimgs = [*obj0.imgs, *obj1.imgs]
            new_w_img = [*obj0.w_img, *obj1.w_img]
            newobs.objs[newid] = Obj(newid, newimgs, new_w_img, [newcenter], [
                                     new_w_center], newdir, new_w_dir, newpose, 128)
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
    car_widt = 0.14
    labellocs, labeldirs, vis3d = vislabel.main(d_bias=-0.15)
    infopath = Path("outputs/MED_loc")
    viser = LocViser(infopath, vis3d)
    # demo = "00102.jpg"
    # info0 = viser.visdet(demo)
    # demo = "00087.jpg"
    # info1 = viser.visdet(demo)
    res = glob.glob("outputs/MED_loc/results/*0.070.npy")
    results = {}
    for f in res:
        results[os.path.basename(f)[0:9]] = np.load(
            f, allow_pickle=True).item()

    demoimgs = [
        "00021.jpg",  # d = -0.1
        "00087.jpg",  # d = -0.15
        "00097.jpg",
        "00102.jpg"]
    
    locinfopath = Path("outputs/MED_loc/locinfo")
    croppath = Path("outputs/MED_loc/crops")
    ob0 = Observe(name="00102.jpg", croppath=croppath,
                  locinfo=locinfopath/"00102.npy", objlist=results["00102.jpg"]["00102.jpg"])
    
    ob1 = Observe(name="00087.jpg", croppath=croppath,
                  locinfo=locinfopath/"00087.npy", objlist=results["00087.jpg"]["00087.jpg"])
    
    ob2 = Observe(name="00097.jpg", croppath=croppath,
                  locinfo=locinfopath/"00097.npy", objlist=results["00097.jpg"]["00097.jpg"])
    
    ob3 = Observe(name="00021.jpg", croppath=croppath,
                  locinfo=locinfopath/"00021.npy", objlist=results["00021.jpg"]["00021.jpg"])
    
    mg = Merger(dist_thresh=car_length)
    newobs = mg.merge(ob0, ob1)
    mergeinfo = newobs.getobjinfo()
    
    viser.visboxes(centers=mergeinfo['locs'],
                   dirs=mergeinfo['dirs'], name=mergeinfo['name'])
    
    info2 = viser.visdet("00097.jpg")
    newobs = mg.merge(ob2, newobs)
    print(newobs)
    
    
    var_thresh = car_widt/2
    # res0 = valuate(info0['locs'], info0['dirs'],
    #                labellocs, labeldirs, distthresh=var_thresh)
    # res1 = valuate(info1['locs'], info1['dirs'],
    #                labellocs, labeldirs, distthresh=var_thresh)
    resmerge = valuate(mergeinfo['locs'], mergeinfo['dirs'],
                    labellocs, labeldirs, distthresh=var_thresh)
    mergeinfo = newobs.getobjinfo()
    res2 = valuate(info2['locs'], info2['dirs'],
                   labellocs, labeldirs, distthresh=var_thresh)
    resmerge = valuate(mergeinfo['locs'], mergeinfo['dirs'],
                       labellocs, labeldirs, distthresh=var_thresh)
    viser.visboxes(centers=mergeinfo['locs'],
                dirs=mergeinfo['dirs'], name=mergeinfo['name'])
    
    # print(ob0)
    # loss_fn_alex = lpips.LPIPS(net='alex')
    # imgfiles0 = sorted(glob.glob("outputs/MED_loc/crops/00087*"))
    # imgfiles1 = sorted(glob.glob("outputs/MED_loc/crops/00102*"))
    # imgls0 = []
    # imgls1 = []
    # for imgf in imgfiles0:
    #     im = cv2.imread(imgf, cv2.IMREAD_ANYCOLOR)
    #     im = cv2.resize(im, (128, 128)).astype(np.float32)
    #     im = 2*(im-im.min())/(im.max()-im.min())-1.0
    #     im = im.transpose(2, 0, 1)
    #     imgls0.append(im)

    # for imgf in imgfiles1:
    #     im = cv2.imread(imgf, cv2.IMREAD_ANYCOLOR)
    #     im = cv2.resize(im, (128, 128)).astype(np.float32)
    #     im = 2*(im-im.min())/(im.max()-im.min())-1.0
    #     im = im.transpose(2, 0, 1)
    #     imgls1.append(im)
    # for id0, name0 in enumerate(imgfiles0):
    #     minscore = np.inf
    #     minname = None
    #     for id1, name1 in enumerate(imgfiles1):
    #         im0 = torch.from_numpy(imgls0[id0][None])
    #         im1 = torch.from_numpy(imgls1[id1][None])
    #         d = loss_fn_alex(im0, im1)
    #         if d < minscore:
    #             minscore = d
    #             minname = name1
    #     print(name0, minname)
