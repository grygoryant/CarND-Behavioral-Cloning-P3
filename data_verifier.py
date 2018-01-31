import os
import sys
import csv

def main(argv):
	path = './data/'
	if len(argv) != 0:
		path = argv[0]

	driving_log_path = path + 'driving_log.csv'
	with open(driving_log_path) as data_file:
		reader = csv.reader(data_file)
		for line in reader:
			for i in range(3):
				file_name = path + 'IMG/' + line[i].split('/')[-1]
				if not os.path.isfile(file_name):
					print('ERROR: File ' + file_name + ' not exists!')

if __name__ == "__main__":
   main(sys.argv[1:])
