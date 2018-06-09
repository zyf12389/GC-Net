# GC-Net
gc-net for stereo matching by using pytorch

《End-to-End Learning of Geometry and Context for Deep Stereo Regression》https://arxiv.org/abs/1703.04309

Files that begin with a read are for extracting data from sceneflow dataset. read_data.py is just like a dataprovider which can be processed by dataloader. 

Param: Batch size in my code was setted 160 (should be should be a multiple of 32 because of the encoder-decoder process)
