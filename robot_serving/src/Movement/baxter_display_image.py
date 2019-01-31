import argparse
import cv2

def baxter_display_image():
	# Setup parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path',
						help='Path to Baxter Expressions Image Folder',
						type=str,
						required=True)
	args = parser.parse_args()
	folder_path = args.path

	print(folder_path)

	img = cv2.imread('./' + folder_path + '/test_img.jpg')
	cv2.imshow('Image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	baxter_display_image()
