import cv2
import os
def flip_images():
	gest_folder = r"G:\Draft\data\train"
	for g_id in os.listdir(gest_folder):
		for i in range(499):
			path = gest_folder+"/"+g_id+"/"+str(i+1)+".jpg"
			new_path = gest_folder+"/"+g_id+"/"+str(i+499+1)+".jpg"
			img = cv2.imread(path, 0)
			img = cv2.flip(img, 1)
			cv2.imwrite(new_path, img)
	gest_folder1 = r"G:\Draft\data\test"
	for g_id in os.listdir(gest_folder1):
		for i in range(499,599):
			path = gest_folder1+"/"+g_id+"/"+str(i+1)+".jpg"
			new_path = gest_folder1+"/"+g_id+"/"+str(i+599+1)+".jpg"
			img = cv2.imread(path, 0)
			img = cv2.flip(img, 1)
			cv2.imwrite(new_path, img)
	gest_folder2 = r"G:\Draft\data\validation"
	for g_id in os.listdir(gest_folder2):
		for i in range(99):
			path = gest_folder2 + "/" + g_id + "/" + str(i + 1) + ".jpg"
			new_path = gest_folder2 + "/" + g_id + "/" + str(i + 99 + 1) + ".jpg"
			img = cv2.imread(path, 0)
			img = cv2.flip(img, 1)
			cv2.imwrite(new_path, img)

flip_images()
