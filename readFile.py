import os
import pickle
# dir = "disparity"
dir_driving='../frame_cleanpass_webp_driving/frames_cleanpass_webp'
dir_flying='../frames_cleanpass_webp_flying/TRAIN'
dir_monkk='../monkaa_frames_cleanpass_webp/frames_cleanpass_webp'
paths=[]
for root, dirs, files in os.walk(dir_driving):
    for file in files:
        paths.append(os.path.join(root,file))
for root, dirs, files in os.walk(dir_flying):
    for file in files:
        paths.append(os.path.join(root,file))
for root, dirs, files in os.walk(dir_monkk):
    for file in files:
        paths.append(os.path.join(root,file))
paths_left=[]
paths_right=[]
for i in range(len(paths)):
	if paths[i].find('left')>-1:
		paths_left.append(paths[i])
	elif paths[i].find('right')>-1:
		paths_right.append(paths[i])
foutl=open('paths_left_train.pkl','wb')
foutr=open('paths_right_train.pkl','wb')
pickle.dump(paths_left,foutl)
pickle.dump(paths_right,foutr)
foutl.close()
foutr.close()
print(len(paths))
