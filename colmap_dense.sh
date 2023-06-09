SFMPATH=outputs/YKL_sfm_gnd/sfm_superpoint+superglue
IMGPATH=datasets/YKL/images
echo $SFMPATH
colmap image_undistorter \
    --image_path $IMGPATH \
    --input_path $SFMPATH\
    --output_path $SFMPATH/dense \
    --output_type COLMAP \
    --max_image_size 2000

colmap patch_match_stereo \
    --workspace_path $SFMPATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
    --workspace_path $SFMPATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $SFMPATH/dense/fused.ply