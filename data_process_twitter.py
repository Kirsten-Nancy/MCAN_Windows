# -*- coding: utf-8 -*-

import re
import os
from PIL import Image
import numpy as np
import csv
import itertools

# data_path = '/home/**/workspace/data/2016_twitter_data'
data_path = 'e:\\nlp_projects\\MCAN_Windows\\twitter_data'

original_train_data = os.path.join(data_path, 'train_original.txt')
original_test_data = os.path.join(data_path, 'test_original.txt')

new_train = os.path.join(data_path, 'train_posts.csv')
new_test = os.path.join(data_path, 'test_posts.csv')

# image_file_list = [os.path.join(data_path, 'train_images/fake_images/'), os.path.join(data_path, 'train_images/real_images/'), os.path.join(data_path, 'test_images/')]
image_file_list = [os.path.join(data_path, 'fake_images/'), os.path.join(data_path, 'real_images/'), os.path.join(data_path, 'test_images/')]


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def read_images(file_list):
    image_list = {} 
    img_num = 0
    for path in file_list:
        for filename in os.listdir(path):
            try:
                img = Image.open(path + filename).convert('RGB')
                img_id = filename.split('.')[0]
                image_list[img_id] = img
                #print('ok')
                img_num += 1
                
            except:
                # print('Images where it is txt file',filename)
                pass
    # print('image list', image_list)
    return image_list#, img_num



def select_image(image_num, image_id_list, image_list):
    for i in range(image_num):
        #print('list:{}'.format(image_id_list))
        # print('Type', type(image_id_list))
        image_id = image_id_list[i].split('.')[0]
        if image_id in image_list:
            #print('Yes, img_id:{}'.format(img_id))
            return image_id
    #f_log.write(line)
    return False            
            
     

def get_max_len(file):

    #Get the maximal length of sentence in dataset

    f = open(file, 'r', encoding = 'UTF-8')
    
    max_post_len = 0
    
    lines = f.readlines()
    post_num = len(lines)
    for i in range(post_num):
        post_content = list(lines[i].split('|')[1].split())
        tmp_len = len("".join(post_content))
        if tmp_len > max_post_len:
            max_post_len = tmp_len
            
    f.close()
    return max_post_len

def get_data(dataset, image_list):
    if dataset == 'train':        
        data_file = new_train
    else:
        data_file = new_test
        
    # f = open(data_file, 'r', encoding = 'UTF-8')
    # lines = f.readlines()

    lines = []

    with open(data_file, "r", encoding="utf8") as f:
        # lines = list(csv.DictReader(f))
        csv_reader = csv.DictReader(f)

        # next(csv_reader, None)

        for row in csv_reader:
            lines.append(row)

    
    data_post_id = []
    data_post_content = []
    data_image = []
    data_label = []   
        
    data_num = len(lines)
    unmatched_num = 0

    # sample_data = lines[:200]

    for line in lines:
        post_id = line['post_id']
        post_content = line['post_text']
        label = line['label']

        if dataset == 'train':      
            if ',' in line['image_id(s)']:
                image_id_list = line['image_id(s)'].split(',')
                img_num = len(image_id_list)
            else:
                image_id_list = [line['image_id(s)']]
                img_num = 1
        
        else:
            image_id_list = [line['image_id']]
            img_num = 1

        if label == 'fake':
            label = '1'
        elif label == 'real':
            label = '0'
        else:                    
            print('The label of this tweet is humor, we dont need it.')
     
        image_id = select_image(img_num, image_id_list, image_list)
            
        if image_id != False:
            image = image_list[image_id]
                    
            data_post_id.append(int(post_id))
            data_post_content.append(post_content)
            data_image.append(image)
            data_label.append(int(label))
            # TODO: Change the label to int -> 0 or 1 for fake and real
            # data_label.append(label)
            # print('Post content: ',post_content)   
            # print('Image ids found', image_id_list)          
        else:
            # print('Image ids missing', image_id_list)
            unmatched_num += 1
            continue
            
    # print('Data length', len(data_post_content))
    # print('Unmatched', unmatched_num)
    data_dic = {'post_id': np.array(data_post_id),
                'post_content': data_post_content,
                'image': data_image,
                'label': np.array(data_label)
                }
 
    # print('Dict content', data_dic)
    return data_dic, data_num-unmatched_num              

# """
# if __name__ == '__main__':
    
#     # fake_num, real_num = select_data(original_train_data, new_train )    
#     #print(fake_num, real_num)
#     #max_len = get_max_len(new_train)
#     #print(max_len)
#     # f_log = open(log, 'w', encoding = 'UTF-8')
    
#     img_list = read_images(image_file_list)
#     #print(img_num)
#     # print('image', len(img_list))
#     train, train_num = get_data('train', img_list)
#     # print(len(train['label']))
#     # print(train)
#     # print('train')
# # """

# weibo_data/
# twitter_data/
# bert-base-chinese/
# log/
# vgg19-dcbb9e9d.pth









