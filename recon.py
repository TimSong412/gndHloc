
from pathlib import Path

from hloc import extract_features, match_features, reconstr, visualization, pairs_from_retrieval

images = Path('datasets/MMW/images_all/')

outputs = Path('outputs/MMW_all/')
sfm_pairs = outputs / 'pairs-netvlad.txt'
sfm_dir = outputs / 'sfm_superpoint+superglue'

retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

model = reconstr.main(sfm_dir, images, sfm_pairs, feature_path, match_path)

