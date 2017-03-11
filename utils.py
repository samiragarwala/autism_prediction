import sys
import os
import h5py
import matplotlib as ml
ml.use("agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def mat2visual(mat, zLocs, filename, valRange='auto'):
	'''
	Visualizes the input numpy matrix and saves it into a file

	Example Usage:
	    data = loadfALFF(4)
	    mat2visual(data, [40,45,50], 'example.png')

	@type   mat         :   3D numpy matrix
	@param  mat         :   3D data matrix to visualize
	@type   zLocs       :   int array
	@param  zLocs       :   Specifies the z positions to slice the mat matrix at
	@type   filename    :   String
	@param  filename    :   Name of the file to save the sliced brains to.
	@type   valRange    :   int tuple or 'auto'
	@param  valRange    :   Specifies the maximum and minimum values of the colorbar used in imshow.
							auto for auto-scaling of the input
	'''
	_,_,c = mat.shape
	plt.close("all")
	plt.figure()
	for i in range(len(zLocs)):
		if zLocs[i]>=c:
			print("An element %d in zLocs is larger than %d" %(zLocs[i],c))
			return
		plt.subplot(1,len(zLocs),i+1)
		plt.title('z='+str(zLocs[i]))
		if type(valRange) is str and valRange=='auto':
			plt.imshow(mat[:,:,zLocs[i]], cmap = "gray", interpolation='none')
		else:
			plt.imshow(mat[:,:,zLocs[i]], vmin=min(valRange), vmax=max(valRange), cmap = "gray", interpolation='none')

	ax = plt.gca()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.1)
	plt.colorbar(cax=cax)

	plt.savefig(filename)