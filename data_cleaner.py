import cv2
import os
import csv
import sys

def show_img(img):
	cv2.imshow('frame', img)
	if cv2.waitKey(0) == ord('r'):
		cv2.destroyAllWindows()
		return True
	cv2.destroyAllWindows()
	return False

def draw_debug_data(img, angle):
	debug_img = img
	font = cv2.FONT_HERSHEY_SIMPLEX
	h,w = debug_img.shape[0:2]
	cv2.line(debug_img,(int(w/2),int(h)),(int(w/2+angle*w/4),int(h/2)),(0,255,0),thickness=3)
	return debug_img

def main(argv):
	path = './data/'
	if len(argv) != 0:
		path = argv[0]

	driving_log_path = path + 'driving_log.csv'
	lines = []
	with open(driving_log_path) as data_file:
		reader = csv.reader(data_file)
		next(reader, None)
		for line in reader:
			lines.append(line)

	lines_to_delete = []

	for line in lines:
		center_name = path + 'IMG/' + line[0].split('/')[-1]
		left_name = path + 'IMG/' + line[1].split('/')[-1]
		right_name = path + 'IMG/' + line[2].split('/')[-1]
		center_image = cv2.imread(center_name)
		left_image = cv2.imread(left_name)
		right_image = cv2.imread(right_name)
		center_angle = float(line[3])
		left_angle = center_angle + 0.2
		right_angle = center_angle - 0.2	

		if show_img(draw_debug_data(center_image, center_angle)):
			print('Line to remove:')
			print(line)
			lines_to_delete.append(line)

	with open(driving_log_path) as data_file, \
		open(path + 'driving_log_edited.csv', 'w') as out_file:

		writer = csv.writer(out_file)

		for row in csv.reader(data_file):
			to_remove = False
			for line in lines_to_delete:
				if line[0] in row or  \
					line[1] in row or \
					line[2] in row:
					to_remove = True
					print('Removing line:')
					print(row)

					for i in range(3):
						file = line[i]
						name = file.split('/')[-1]
						try:
							print('Removing file:')
							print(name)
							os.remove(path + 'IMG/' + name)
						except OSError:
							pass

			if not to_remove:
				writer.writerow(row)

if __name__ == "__main__":
   main(sys.argv[1:])



