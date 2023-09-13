# -*- coding: utf-8 -*-

# https://github.com/wangzhuang1911/Weibo-dataset -> Link to how the dataset lines are divided
import re
import os
from PIL import Image
import numpy as np

data_path = 'e:\\nlp_projects\\MCAN_Windows\\weibo_data'

original_train_data = os.path.join(data_path, 'train_original.txt')
original_test_data = os.path.join(data_path, 'test_original.txt')

# new_train = os.path.join(data_path, 'train.txt')
# new_test = os.path.join(data_path, 'test.txt')

new_train = os.path.join(data_path, 'train_rumor.txt')
new_test = os.path.join(data_path, 'test_rumor.txt')

image_file_list = [os.path.join(data_path, 'rumor_images/'), os.path.join(data_path, 'nonrumor_images/')]


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def read_images(file_list):
    image_list = {} 
    #img_num = 0
    for path in file_list:
        for filename in os.listdir(path):
            try:
                img = Image.open(path + filename).convert('RGB')
                img_id = filename.split('.')[0]
                image_list[img_id] = img
                #print('ok')
                #img_num += 1
            except:
                print(filename)
    return image_list#, img_num



def select_image(image_num, image_id_list, image_list):
    for i in range(image_num):
        #print('list:{}'.format(image_id_list))
        image_id = image_id_list[i].split('.')[0]
        # print('image id in select image', image_id)
        if image_id in image_list:
            #print('Yes, img_id:{}'.format(img_id))
            return image_id
    #f_log.write(line)
    return False            
            

def select_data(twitter_original_data, twitter_selected_data):
    """
    select features that we need from original data
    """
    f_old = open(twitter_original_data, 'r', encoding = 'UTF-8')
    f_new = open(twitter_selected_data, 'w', encoding = 'UTF-8')
    
    fake_count = 0
    real_count = 0
    
    lines = f_old.readlines()
    for i, l in enumerate(lines):
        if i == 0:
            continue
        else:
            postId = l.split()[0]
            label = l.split()[-1]
            #print('postID:{}, label:{}'.format(postId, label))
                
            imageId_list = l.split('	')[4]
            postText = l.split('	')[1]
            
            clean_postText = re.sub(r"(http|https)((\W+)(\w+)(\W+)(\w*)(\W+)(\w*)|(\W+)(\w+)(\W+)|(\W+))","",postText)
            #print('clean_postText:{}'.format(clean_postText))
                
            f_new.write(postId + '|' + clean_postText + '|' + imageId_list + '|' + label + '\n')
                    
    f_old.close()
    f_new.close()
    
    return fake_count, real_count
     

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
        
    f = open(data_file, 'r', encoding = 'UTF-8')
    lines = f.readlines()
        
    data_post_id = []
    data_post_content = []
    data_image = []
    data_label = []   
        
    data_num = len(lines)
    unmatched_num = 0
    
    # print('First 3', lines[:3])
    # for line in lines:
    new_lines = []
    for i in range(0, len(lines), 3):
        group = lines[i:i+3]
        if not group:
            break
        # merged_line = ''.join(group).replace('/n','')
        merged_line = ''.join(group)
        new_lines.append(merged_line)
    
    # print('Merged lines', new_lines[:3])

    for index in range(0, len(new_lines)):
        # print('Line', new_lines[index].split('/n'))
        # print('Check new line', '\n' in new_lines[index])
        # break
        current_line = new_lines[index].split('\n')
        # print('Current line', current_line)
        post_id = current_line[0].split('|')[0]
        post_content = current_line[2]
        label = current_line[0].split('|')[5]
        
        image_id_list = current_line[1].split('|')[:-1]
        #print(image_id_list)
        new_image_id_list = []
        for image in image_id_list:
            img_id = image.split('/')[4]
            new_image_id_list.append(img_id)

        # print('new_image_id_list', new_image_id_list)
        # break
        if label == 'true':
            label = '1'
        elif label == 'real':
            label = '0'
        else:                    
            print('The label of this tweet is humor, we donnot need it.')

        img_num = len(new_image_id_list)
        image_id = select_image(img_num, new_image_id_list, image_list)

        # img_num = len(image_id_list)
        # image_id = select_image(img_num, image_id_list, image_list)

        # print('Post ID', post_id)
        # print('Post content', post_content)
        # print('Post label', label)

        # print('image_id_list', image_id_list)
        # break
        # print('image id', image_id)
        # break
        if image_id != False:
            image = image_list[image_id]
                    
            data_post_id.append(int(post_id))
            data_post_content.append(post_content)
            data_image.append(image)
            data_label.append(int(label))
                    
        else:
            unmatched_num += 1
            continue
    print('Number of unmatched', unmatched_num)
    f.close()
    
    data_dic = {'post_id': np.array(data_post_id),
                'post_content': data_post_content,
                'image': data_image,
                'label': np.array(data_label)
            }
    # print('data dict', data_dic)
    return data_dic, data_num-unmatched_num              

"""       
if __name__ == '__main__':
    
    #fake_num, real_num = select_data(original_train_data, new_train)    
    #print(fake_num, real_num)
    #max_len = get_max_len(new_train)
    #print(max_len)
    # f_log = open(log, 'w', encoding = 'UTF-8')
    
    img_list = read_images(image_file_list)
    # print('image list', img_list)
    print(len(img_list))
    train, train_num = get_data('train', img_list)
    #print(train_num)
"""










