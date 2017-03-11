import numpy as np
import h5py
import os
import sys

#Sys.args- 1 for numpy locations, 2 for hdf5 location, 3 for numpy test location (same as 1)

#Converts multiple numpy files to multiple hdf5 files. 
#The location of numpy files and destination OF hdf5 
#files to be mentioned.
def convert_to_hdf5(): 
	path =sys.argv[1]   
	files = os.listdir(path)   
	for name in files: 
		if(name==".DS_Store"):
			continue
		BASE_PATH = sys.argv[2]  
		name_wo_ext = os.path.splitext(name)[0]
		file_name=os.path.splitext(os.path.join(BASE_PATH, name))[0]
		a = np.load(os.path.join(BASE_PATH, name), mmap_mode=None, allow_pickle=False, fix_imports=True)
		h5f = h5py.File(file_name+".hdf5", 'w')
		h5f.create_dataset(name_wo_ext, data=a)
		h5f.close()

#For Testing Purposes
# def rand_numpy(): 
# 	a=np.random.rand(2,4)
# 	BASE_PATH = sys.argv[3]
# 	np.save(os.path.join(BASE_PATH, "bat"), a , allow_pickle=False, fix_imports=True)
# 	print(a)

# def hdf5_test():
# 	hf=h5py.File(str(sys.argv[2])+'bat.hdf5','r')
# 	data = hf.get('bat')
# 	np_data = np.array(data)
# 	print(np_data)


if __name__ == "__main__": 
	rand_numpy()
	convert_to_hdf5()
	hdf5_test()