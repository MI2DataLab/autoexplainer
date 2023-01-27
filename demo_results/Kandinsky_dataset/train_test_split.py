import os
import shutil
from tqdm import tqdm
import random

def moveDatasetsToImageFolders(folder_with_datasets, output_dir, split_ratio):
    random.seed(42)
    os.makedirs(output_dir, exist_ok=True)

    dataset_folders = os.listdir(folder_with_datasets)
    for dataset_folder in dataset_folders:
        print('Processing dataset: ' + dataset_folder)
        class_names = os.listdir(folder_with_datasets + '/' + dataset_folder)
        for class_name in class_names:
            imgs_list = os.listdir(folder_with_datasets + '/' + dataset_folder + '/' + class_name)

            random_img_ids = random.sample(range(len(imgs_list)), int(len(imgs_list) * split_ratio))
            train_imgs_list = [imgs_list[i] for i in random_img_ids]
            test_imgs_list = [imgs_list[i] for i in range(len(imgs_list)) if i not in random_img_ids]

            os.makedirs(output_dir + '/train/' + dataset_folder + '_' + class_name, exist_ok=True)
            os.makedirs(output_dir + '/test/' + dataset_folder + '_' + class_name, exist_ok=True)

            for img_name in tqdm(train_imgs_list, desc='Processing (train) class: ' + class_name):
                shutil.copy(folder_with_datasets + '/' + dataset_folder + '/' + class_name + '/' + img_name,
                            output_dir + '/train/' + dataset_folder + '_' + class_name + '/' + img_name)
            for img_name in tqdm(test_imgs_list, desc='Processing (test) class: ' + class_name):
                shutil.copy(folder_with_datasets + '/' + dataset_folder + '/' + class_name + '/' + img_name,
                            output_dir + '/test/' + dataset_folder + '_' + class_name + '/' + img_name)

if __name__ == '__main__':
    split_ratio = 0.9
    root_dir = 'data/'
    output_dir_name = root_dir + 'all_datasets'
    input_dir_with_downloaded_datasets = root_dir + 'data_downloaded'

    moveDatasetsToImageFolders(folder_with_datasets = input_dir_with_downloaded_datasets, output_dir = output_dir_name,
                               split_ratio = split_ratio)
    input_dir_with_generated_datasets = root_dir + 'data_generated'
    moveDatasetsToImageFolders(folder_with_datasets = input_dir_with_generated_datasets, output_dir = output_dir_name,
                               split_ratio = split_ratio)