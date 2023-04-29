
from pathlib import Path
from hloc import logger

from hloc import extract_features, match_features, reconstr, pairs_from_retrieval, pairs_from_covisibility
import threading


def genmask(imagepth: Path, outdir: Path="dbmasks"):

    pass

def genGlobalMatch(images, retrieval_conf, sfm_pairs, outputs):
    logger.info("GLOBAL BEGIN")
    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=10)
    logger.info("GLOBAL DONE")
    return

def genLocalFeat(feature_conf, images, outputs):
    logger.info("LOCAL BEGIN")
    feature_path = extract_features.main(feature_conf, images, outputs)
    logger.info("LOCAL DONE")
    return

def main():
    images = Path('datasets/MMW/images_all/')

    outputs = Path('outputs/MMW_gndall/')
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
    # pairs_from_covisibility.main(model, sfm_pairs, num_matched=20)

if __name__ == "__main__":
    main()