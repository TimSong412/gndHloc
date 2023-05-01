from pathlib import Path
from pprint import pformat
import os, time, cv2
import numpy as np
import pycolmap

from hloc import extract_features_inc, match_features_inc, pairs_from_retrieval_inc
from hloc import localize_sfm_inc, visualization

def main():
    dataset = Path('datasets/MMW/')  # change this if your dataset is somewhere else
    images = dataset / 'query'
    sfmpath = Path("outputs/MMW_gndsfm")
    outputs = Path('outputs/MMW_loc_inc/')  # where everything will be saved
    sfm_pairs = sfmpath / "pairs-netvlad.txt"  # top 20 most covisible in SIFT model 
    loc_pairs = outputs / "loc_pairs.txt"  # top 20 retrieved by NetVLAD
    
    results = outputs / 'MMW_hloc_superpoint+superglue_netvlad20.txt'  # the result file

    # list the standard configurations available
    print(f'Configs for feature extractors:\n{pformat(extract_features_inc.confs)}')
    print(f'Configs for feature matchers:\n{pformat(match_features_inc.confs)}')

    # pick one of the configurations for image retrieval, local feature extraction, and matching
    # you can also simply write your own here!
    retrieval_conf = extract_features_inc.confs['netvlad']
    feature_conf = extract_features_inc.confs['superpoint_aachen']
    matcher_conf = match_features_inc.confs['superglue-fast']
    db_descriptors = sfmpath / "global-feats-netvlad.h5"
    db_model = sfmpath / "sfm_superpoint+superglue"
    reconstruction = db_model
    feature_ref = sfmpath / "feats-superpoint-n4096-r1024.h5"
        
    
    

    local_extor = extract_features_inc.FeatureExtractor(conf=feature_conf)
    global_extor = extract_features_inc.FeatureExtractor(conf=retrieval_conf)
    pairmatcher = pairs_from_retrieval_inc.PairRetriever(num_matched=3, db_model=db_model, db_descriptors=db_descriptors)
    featmatcher = match_features_inc.FeatureMatcher(matcher_conf, features_ref=feature_ref)
    locer = localize_sfm_inc.Localizer(reconstruction)

    imgname = "1109_MMW_DJI_0005_00087.jpg"
    img = cv2.imread((images / imgname).__str__(), cv2.IMREAD_ANYCOLOR)
    query_cam = [(imgname, pycolmap.Camera('SIMPLE_RADIAL', int(1920), int(1080), np.array([1209.676922100206, 960.0, 540.0, 0.004254746417600625])))]

    t0 = time.time()
    '''
    query local features
    '''
    query_local_feats = local_extor.imgextract(img, name=imgname)

    # local_feats = extract_features_inc.main(feature_conf, Path("datasets/MMW/images_all"), outputs) 

    t1 = time.time()

    '''
    query global descriptors
    '''
    # every query image has a descriptor
    
    query_global = global_extor.imgextract(img, name=imgname)

    # descriptors = extract_features_inc.main(retrieval_conf, images, outputs)

    t2 = time.time()

    '''
    global localize (image pair)
    '''    
    # pairs_from_retrieval_inc.main(descriptors, loc_pairs, num_matched=3,  query_prefix=None, db_model=db_model, db_descriptors=db_descriptors)
    imgpairs = pairmatcher.match(query_global, imgname)
    # reconstruction = reference_sfm
    
    t3 = time.time()
    
    '''
    match local feature
    '''    
    # match_path = outputs / "global_localfeats_match.h5"
    # loc_matches = match_features_inc.main(matcher_conf, loc_pairs, local_feats, outputs, matches=match_path, features_ref=feature_ref)
    print("FEATMATCH DONE")
    query_featmatch = featmatcher.match(imgpairs, feature_q=query_local_feats)
    t4 = time.time()


    '''
    localization
    '''
    # localize_sfm_inc.main(
    #     reconstruction,    
    #     dataset/ "qimg_new.txt", # dataset / 'queries/*_time_queries_with_intrinsics.txt',
    #     loc_pairs,
    #     local_feats,
    #     loc_matches,
    #     results,
    #     covisibility_clustering=False)  # not required with SuperPoint+SuperGlue
    
    pose = locer.loc(queries=query_cam, retrieval=imgpairs, features=query_local_feats, matches=query_featmatch)
    print("pose= ", pose)
    t5 = time.time()
    print("localfeats t1 = ", t1-t0)
    print("globaldesc t2 = ", t2-t1)
    print("imgretriev t3 = ", t3-t2)
    print("featmatch  t4 = ", t4-t3)
    print("pnp-loc    t5 = ", t5-t4)

    visualization.visualize_loc(
        results, images, reconstruction, n=20, top_k_db=1, prefix=None, seed=2, db_image_dir=dataset/"images")


    
if __name__ == "__main__":
    main()