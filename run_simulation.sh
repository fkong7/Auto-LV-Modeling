# Generate Simulation Ready File for svFSI

# Id of the phase to generate volume mesh
# This information can be obtained from the 
# last line of the shell output of run_volmesh.sh
phase_id=0

# Path to the registered surface meshes
input_dir=./04-SurfReg/BD9702
input_dir=/Users/fanweikong/Documents/Modeling/HeartDeepFFD/output/bspline_ctrl_pts_16_uniform_fit_amp0.1_ctrl100_lr1e-3_full_DownFrom16AddDiff_fullyConnectedGraph_MiniAug_debug_featPoolFix/BD9702_left/processed
input_dir=/Users/fanweikong/Documents/Modeling/HeartDeepFFD/output/bspline_ctrl_pts_16_uniform_fit_amp0.1_ctrl100_lr1e-3_full_DownFrom16AddDiff_fullyConnectedGraph_MiniAug_debug_featPoolFix/BD9702_right/processed
# Path to the outputed volume meshes
output_dir=./05-VolMesh/BD9702
output_dir=/Users/fanweikong/Documents/Modeling/HeartDeepFFD/output/bspline_ctrl_pts_16_uniform_fit_amp0.1_ctrl100_lr1e-3_full_DownFrom16AddDiff_fullyConnectedGraph_MiniAug_debug_featPoolFix/BD9702_left/processed/motion
output_dir=/Users/fanweikong/Documents/Modeling/HeartDeepFFD/output/bspline_ctrl_pts_16_uniform_fit_amp0.1_ctrl100_lr1e-3_full_DownFrom16AddDiff_fullyConnectedGraph_MiniAug_debug_featPoolFix/BD9702_right/processed
# Number of interpolations between adjacent phases
num=99
# Number of cardiac cycles
cyc=1
# Cycle duriation in seconds
period=1.25
period=0.5

# Volumetric Meshing using SimVascular
python ./Modeling/svfsi/interpolation.py \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --num_interpolation $num \
    --num_cycle $cyc \
    --duration $period \
    --phase $phase_id

