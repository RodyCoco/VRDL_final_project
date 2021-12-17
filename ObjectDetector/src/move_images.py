import os
import shutil

ORIGINAL_PATH = '../images/train/'
classes = ['ALB', 'BET', 'DOL', 'LAG', 'SHARK', 'YFT', 'NoF', 'OTHER']

for c in classes:
	source_dir = os.path.join(ORIGINAL_PATH, c)
	file_names = os.listdir(source_dir)
	for file_name in file_names:
		shutil.move(os.path.join(source_dir, file_name), ORIGINAL_PATH)
