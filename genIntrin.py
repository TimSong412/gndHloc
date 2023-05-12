import glob
import hloc.utils.read_write_model as rw
from pathlib import Path
import os

dbpath = Path("datasets/YKL/")
modelpath = Path("outputs/YKL_q_sfm_gnd/sfm_superpoint+superglue")
outfile = dbpath/"qimgs.txt"
qimg = glob.glob((dbpath/"query/*").__str__())
qimg = [os.path.basename(q) for q in qimg]
print(qimg)
imgs = rw.read_images_binary(modelpath/"images.bin")
cams = rw.read_cameras_binary(modelpath/"cameras.bin")
imgkname = {}
for im in imgs.values():
    imgkname[im.name] = im

f = open(outfile, 'w')
for q in qimg:
    camid = imgkname[q].camera_id
    caminfo = cams[camid]
    f.write(f"{q} SIMPLE_RADIAL {caminfo.width} {caminfo.height} {caminfo.params[0]} {caminfo.params[1]} {caminfo.params[2]} {caminfo.params[3]}\n")