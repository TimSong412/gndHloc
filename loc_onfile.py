from pathlib import Path
from pprint import pformat
import os, time

from hloc import extract_features_inc, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization

def main():
    dataset = Path('datasets/MMW/')  # change this if your dataset is somewhere else
    images = dataset / 'query'
    sfmpath = Path("outputs/MMW_sfm_gnd")
    outputs = Path('outputs/MMW_loc_file/')  # where everything will be saved
    sfm_pairs = sfmpath / "pairs-netvlad.txt"  # top 20 most covisible in SIFT model 
    loc_pairs = outputs / "loc_pairs.txt"  # top 20 retrieved by NetVLAD
    
    results = outputs / 'MMW_hloc_superpoint+superglue_netvlad20.txt'  # the result file

    # list the standard configurations available
    print(f'Configs for feature extractors:\n{pformat(extract_features_inc.confs)}')
    print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

    # pick one of the configurations for image retrieval, local feature extraction, and matching
    # you can also simply write your own here!
    retrieval_conf = extract_features_inc.confs['netvlad']
    feature_conf = extract_features_inc.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    '''
    query local feature
    '''
    local_feats = extract_features_inc.main(feature_conf, Path("datasets/MMW/images_all"), outputs) 

    t1 = time.time()

    '''
    query global descriptors
    '''
    # every query image has a descriptor
    descriptors = extract_features_inc.main(retrieval_conf, images, outputs)
    db_descriptors = sfmpath / "global-feats-netvlad.h5"
    db_model = sfmpath / "sfm_superpoint+superglue"
    t2 = time.time()
    '''
    global localize (image pair)
    '''
    pairs_from_retrieval.main(descriptors, loc_pairs, num_matched=3,  query_prefix=None, db_model=db_model, db_descriptors=db_descriptors)

    # reconstruction = reference_sfm
    reconstruction = db_model
    t3 = time.time()
    feature_ref = sfmpath / "feats-superpoint-n4096-r1024.h5"
    '''
    match local feature
    '''
    match_path = outputs / "global_localfeats_match.h5"
    loc_matches = match_features.main(matcher_conf, loc_pairs, local_feats, outputs, matches=match_path, features_ref=feature_ref)
    t4 = time.time()
    localize_sfm.main(
        reconstruction,    
        dataset/ "qimg_new.txt", # dataset / 'queries/*_time_queries_with_intrinsics.txt',
        loc_pairs,
        local_feats,
        loc_matches,
        results,
        covisibility_clustering=False)  # not required with SuperPoint+SuperGlue
    t5 = time.time()
    visualization.visualize_loc(
        results, images, reconstruction, n=20, top_k_db=1, prefix=None, seed=2, db_image_dir=dataset/"images")


    
    print("t2 = ", t2-t1)
    print("t3 = ", t3-t2)
    print("t4 = ", t4-t3)
    print("t5 = ", t5-t4)

if __name__ == "__main__":
    main()