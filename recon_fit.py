
from pathlib import Path
from hloc import logger

from hloc import extract_features, match_features, reconstr, pairs_from_retrieval, pairs_from_covisibility
import threading
from planefit import fitgndplane
from Givens import genGivens
import hloc.utils.read_write_model as rw
import numpy as np


def genmask(imagepth: Path, outdir: Path="dbimg_masks") -> Path:
    output = imagepth.parent / outdir
    print("maskout= ", output)
    pass

def genGlobalMatch(images, retrieval_conf, sfm_pairs, outputs):
    logger.info("GLOBAL BEGIN")
    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=20)
    logger.info("GLOBAL DONE")
    return

def genLocalFeat(feature_conf, images, outputs):
    logger.info("LOCAL BEGIN")
    feature_path = extract_features.main(feature_conf, images, outputs)
    logger.info("LOCAL DONE")
    return

def main():
    dataset = Path("datasets/MED")
    # images = Path('datasets/MMW/images')
    images = dataset/"images"

    # genmask(imagepth=images, )

    outputs = Path('outputs/MED_sfm_gnd/')
    sfm_pairs = outputs / 'pairs-netvlad.txt'
    sfm_dir = outputs / 'sfm_superpoint+superglue'

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']
    
    feature_path = extract_features.main(feature_conf, images, outputs)
    # local_t = threading.Thread(target=genLocalFeat, args=[feature_conf, images, outputs])
    # local_t.start()
    # local_t.join()
    logger.info("LOCAL JOIN")
    global_t = threading.Thread(target=genGlobalMatch, args=[images, retrieval_conf, sfm_pairs, outputs])
    global_t.start()

    global_t.join()
    logger.info("GLOBAL JOIN")

    print("FEATURE PATH= ", feature_path)

    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)
    logger.info("FEAT MATCH DONE")
    
    
    model = reconstr.main(sfm_dir, images, sfm_pairs, feature_path, match_path)
    # print("model= ", model)
    camdata = rw.read_cameras_binary(sfm_dir/"cameras.bin")
    imgdata = rw.read_images_binary(sfm_dir/"images.bin")
    pts3data = rw.read_points3D_binary(sfm_dir/"points3D.bin")
    

    [a, b, c, d], inliers, inlierpts = fitgndplane(
        cams=camdata, imgs=imgdata, pts3d=pts3data, maskpth=dataset / "masks")
    
    GVtrans = np.eye(4)
    GVtrans[0:3, 0:3] = genGivens(np.array([a, b, c]))
    GVtrans[2, 3]=d
    # a, b, c, d = GVtrans@np.array([a, b, c, d])
    # a, b, c, d = 0, 0, 1, 0
    gndinfo = {'abcd': [a, b, c, d], 'GVtrans': GVtrans}
    np.save(outputs/"gndinfo.npy", gndinfo)

if __name__ == "__main__":
    main()