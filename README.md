# GC-Net
gc-net for stereo matching by using pytorch

《End-to-End Learning of Geometry and Context for Deep Stereo Regression》https://arxiv.org/abs/1703.04309

Requirements:
pytorch0.3.1
python3.5

Files that begin with a read are for extracting data from sceneflow dataset. read_data.py is just like a dataprovider which can be processed by dataloader. 

Max disparity in my code was set 160 because of the limitation of GPU(1080Ti) memory(must be a multiple of 32 because of the encoder-decoder process). Other parameters are basically consistent with those set in the paper.
