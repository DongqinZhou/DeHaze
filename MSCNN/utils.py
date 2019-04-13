from scipy.io import loadmat
import glob
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def resize_img(x_img_list, y_img_list, z_img_list, name):
	final_size = 240
	
	x_image_list = []
	y_image_list = []
	z_image_list= []
	
	for image in x_img_list:
		# print image		
		img = cv2.imread(image)
		w = img.shape[1]
		h = img.shape[0]
		ar = float(w)/float(h)
		if w<h:
			new_w = final_size
			new_h = int(new_w/ar)
			a = new_h - final_size
			resize_img = cv2.resize(img, dsize=(new_w, new_h))
			final_image = resize_img[a/2:a/2+final_size,:]
		elif w>h:
			new_h =final_size
			new_w = int(new_h*ar)
			a = new_w - final_size
			resize_img = cv2.resize(img,dsize=(new_w, new_h))
			final_image = resize_img[:,a/2:a/2+final_size ]
		else:
			resize_img = cv2.resize(img,dsize=(final_size, final_size))
			final_image = resize_img
		
		x_image_list.append(final_image)


	for image in y_img_list:
		# print image		
		img = cv2.imread(image, 0) # Opencv by defaults load an grayscale as an bgr image, with all three channels have same values. 
		w = img.shape[1]			# to load specifically 1 channel, we have to mention that '0'.
		h = img.shape[0]
		ar = float(w)/float(h)
		if w<h:
			new_w = final_size
			new_h = int(new_w/ar)
			a = new_h - final_size
			resize_img = cv2.resize(img, dsize=(new_w, new_h))
			final_image = resize_img[a/2:a/2+final_size,:]
		elif w>h:
			new_h =final_size
			new_w = int(new_h*ar)
			a = new_w - final_size
			resize_img = cv2.resize(img,dsize=(new_w, new_h))
			final_image = resize_img[:,a/2:a/2+final_size ]
		else:
			resize_img = cv2.resize(img,dsize=(final_size, final_size))
			final_image = resize_img
		
		y_image_list.append(final_image)

	for image in z_img_list:
		# print image		
		img = cv2.imread(image)
		w = img.shape[1]
		h = img.shape[0]
		ar = float(w)/float(h)
		if w<h:
			new_w = final_size
			new_h = int(new_w/ar)
			a = new_h - final_size
			resize_img = cv2.resize(img, dsize=(new_w, new_h))
			final_image = resize_img[a/2:a/2+final_size,:]
		elif w>h:
			new_h =final_size
			new_w = int(new_h*ar)
			a = new_w - final_size
			resize_img = cv2.resize(img,dsize=(new_w, new_h))
			final_image = resize_img[:,a/2:a/2+final_size ]
		else:
			resize_img = cv2.resize(img,dsize=(final_size, final_size))
			final_image = resize_img
		
		z_image_list.append(final_image)
	
	npy = []
	for i in range(len(x_image_list)):
		pair = [x_image_list[i], z_image_list[i]]
		npy.append(pair)

	npy = np.array(npy)
	print npy.shape
	trans_npy = np.array(y_image_list)
	trans_npy.resize((trans_npy.shape[0], trans_npy.shape[1], trans_npy.shape[2], 1))
	print trans_npy.shape
	np.save("/media/mnt/dehaze/data/ChinaMM18dehaze/"+name+"_haze_clear.npy", npy)
	np.save("/media/mnt/dehaze/data/ChinaMM18dehaze/"+name+"_trans.npy", trans_npy)

def get_airlight(hzimg,transMap):
	airlight = np.zeros(hzimg.shape)
	kernel = np.ones((15,15),np.uint8)
	for i in range(3):
		img = cv2.erode(hzimg[:,:,i],kernel,iterations = 1)
		airlight[:,:,i] = np.amax(img)
	return airlight

def clearImg(hzimg, transMap):
	airlight = get_airlight(hzimg, transMap)
	clearImg = np.zeros(hzimg.shape)
	transMap = transMap.reshape((transMap.shape[0], transMap.shape[1]))
	constant_matrix = np.ones_like(transMap)*0.1
	clearImg[:,:,0] = (hzimg[:,:,0]-airlight[:,:,0])/np.maximum(constant_matrix, transMap) + airlight[:,:,0]
	clearImg[:,:,1] = (hzimg[:,:,1]-airlight[:,:,1])/np.maximum(constant_matrix, transMap) + airlight[:,:,1]
	clearImg[:,:,2] = (hzimg[:,:,2]-airlight[:,:,2])/np.maximum(constant_matrix, transMap) + airlight[:,:,2]
	clearImg[clearImg<0.0]=0.0
	clearImg[clearImg>1.0]=1.0	
	return clearImg

def test_npy(trans, hazy):
	hazy =  np.load(hazy)
	trans = np.load(trans)
	for i in range(len(trans)):
		plt.imshow(hazy[i][1])
		plt.show()
		plt.imshow(hazy[i][0])
		plt.show()
		plt.imshow(trans[i][:,:,0])
		plt.show()

def main():
	# path = "/home/hitech/Downloads/ChinaMM18dehaze/train/"
	path = "/media/mnt/dehaze/data/ChinaMM18dehaze/train/"

	clear_imgs = glob.glob(path+"clear/*.png")
	trans_imgs = []
	haze_imgs = []

	print len(clear_imgs)

	for img in clear_imgs:
		trans_imgs.append(path+"trans/"+img[51:-4]+"_8.png")
		haze_imgs.append(glob.glob(path+"hazy/"+img[51:-4]+"_8_*.png")[0])
		# trans_imgs.append(path+"trans/"+img[51:-4]+"_8.png")
		# haze_imgs.append(glob.glob(path+"hazy/"+img[51:-4]+"_8_*.png")[0])

	# for c,t,h in zip(clear_imgs, trans_imgs, haze_imgs):
	# 	cv2.imshow("clear", cv2.imread(c))
	# 	cv2.imshow("trans", cv2.imread(t))
	# 	cv2.imshow("hazy", cv2.imread(h))
	# 	k = cv2.waitKey(0)

	# resize_img(haze_imgs[:1000], trans_imgs[:1000], clear_imgs[:1000], "train")
	# resize_img(haze_imgs[1000:], trans_imgs[1000:], clear_imgs[1000:], "val")

	test_npy("/media/mnt/dehaze/data/ChinaMM18dehaze/train_trans.npy","/media/mnt/dehaze/data/ChinaMM18dehaze/train_haze_clear.npy")

if __name__ == "__main__":
	main()