from pathlib import Path
from pprint import pformat
import os, time

from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization

dataset = Path('datasets/MMW/')  # change this if your dataset is somewhere else
images = dataset / 'query'

outputs = Path('outputs/MMW_loc_sfm/')  # where everything will be saved
sfm_pairs = outputs / 'pairs-db-covis20.txt'  # top 20 most covisible in SIFT model 
loc_pairs = outputs / "loc_pairs.txt"  # top 20 retrieved by NetVLAD
reference_sfm = outputs /"sfm_superpoint+superglue"  # the SfM model we will build
results = outputs / 'MMW_hloc_superpoint+superglue_netvlad20.txt'  # the result file

# list the standard configurations available
print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

# pick one of the configurations for image retrieval, local feature extraction, and matching
# you can also simply write your own here!
retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

features = extract_features.main(feature_conf, Path("datasets/MMW/images_all"), outputs)
print("features= ", features)
# colmap_from_nvm.main(
#     dataset / '3D-models/MMW_cvpr2018_db.nvm',
#     dataset / '3D-models/database_intrinsics.txt',
#     dataset / 'MMW.db',
#     outputs / 'sfm_sift')

qimglist = [os.path.basename(i.__str__()) for i in images.glob("*")]

pairs_from_covisibility.main(
    "outputs/MMW_sfm/sfm_superpoint+superglue", sfm_pairs, num_matched=10)#, qimglist=qimglist)

print("COVIS_MATCH DONE")

sfm_matches = match_features.main(matcher_conf, sfm_pairs, Path("outputs/MMW_sfm/feats-superpoint-n4096-r1024.h5"), outputs, matches=Path("outputs/MMW_loc_sfm/feats-superpoint-n4096-r1024_matches-superglue_pairs-db-covis20.h5"))

print("SFM_MATCHES DONE")
t0 = time.time()
# db update new feature 3D keypoints
# reconstruction = triangulation.main(
#     reference_sfm,
#     Path("outputs/MMW_sfm/sfm_superpoint+superglue"),
#     images,
#     sfm_pairs,
#     features,
#     # Path("outputs/MMW_all/feats-superpoint-n4096-r1024.h5"),
#     sfm_matches)
t1 = time.time()

# every image has a descriptor
descriptors = extract_features.main(retrieval_conf, images, outputs)
db_descriptors = Path("outputs/MMW_sfm/global-feats-netvlad.h5")
t2 = time.time()
'''
global pair
'''
pairs_from_retrieval.main(descriptors, loc_pairs, num_matched=20,  query_prefix=None, db_model=Path("outputs/MMW_sfm/sfm_superpoint+superglue"), db_descriptors=db_descriptors)

# reconstruction = reference_sfm
reconstruction = Path("outputs/MMW_sfm/sfm_superpoint+superglue")
t3 = time.time()
feature_ref = Path("outputs/MMW_sfm/feats-superpoint-n4096-r1024.h5")
'''
'''
loc_matches = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs, features_ref=feature_ref)
t4 = time.time()
localize_sfm.main(
    reconstruction,    
    dataset/ "qimg.txt", # dataset / 'queries/*_time_queries_with_intrinsics.txt',
    loc_pairs,
    features,
    loc_matches,
    results,
    covisibility_clustering=False)  # not required with SuperPoint+SuperGlue
t5 = time.time()
visualization.visualize_loc(
    results, images, reconstruction, n=20, top_k_db=1, prefix=None, seed=2, db_image_dir=dataset/"images")


print("t1 = ", t1-t0)
print("t2 = ", t2-t1)
print("t3 = ", t3-t2)
print("t4 = ", t4-t3)
print("t5 = ", t5-t4)