import torch.nn.functional as F
import numpy as np
import torch
from scipy.fftpack import fft,dct
from torch.utils.data import  RandomSampler, SequentialSampler, Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertModel, get_linear_schedule_with_warmup


def process_dct_img(img):
    img = img.numpy() #size = [1, 224, 224]
    height = img.shape[1]
    width = img.shape[2]
    #print('height:{}'.format(height))
    N = 8 
    step = int(height/N) #28

    dct_img = np.zeros((1, N*N, step*step, 1), dtype=np.float32) #[1,64,784,1]
    fft_img = np.zeros((1, N*N, step*step, 1))
    #print('dct_img:{}'.format(dct_img.shape))
    
    i = 0
    for row in np.arange(0, height, step):
        for col in np.arange(0, width, step):
            block = np.array(img[:, row:(row+step), col:(col+step)], dtype=np.float32)
            #print('block:{}'.format(block.shape))
            block1 = block.reshape(-1, step*step, 1) #[batch_size,784,1]
            dct_img[:, i,:,:] = dct(block1) #[batch_size, 64, 784, 1]

            i += 1

    #for i in range(64):
    fft_img[:,:,:,:] = fft(dct_img[:,:,:,:]).real #[batch_size,64, 784,1]
    
    fft_img = torch.from_numpy(fft_img).float() #[batch_size, 64, 784, 1]
    new_img = F.interpolate(fft_img, size=[250,1]) #[batch_size, 64, 250, 1]
    new_img = new_img.squeeze(0).squeeze(-1) #torch.size = [64, 250]
    
    return new_img   


class ModifiedDataset(Dataset):
    def __init__(self, data, VOCAB, max_sen_len, transform_vgg=None, transform_dct=None):
        # super(Dataset, self).__init__()
        # super().__init__()
        super(ModifiedDataset, self).__init__()
        
        self.transform_vgg = transform_vgg
        self.transform_dct = transform_dct
        self.tokenizer = BertTokenizer.from_pretrained(VOCAB, local_files_only=True)
        self.max_sen_len = max_sen_len
        
        self.post_id = torch.from_numpy(data['post_id'])
        self.tweet_content = data['post_content']
        #self.image = list(self.transform(data['image']))
        self.image = list(data['image'])
        self.label = torch.from_numpy(data['label']) #type:int
        
    def __getitem__(self, idx):
        
        content = str(self.tweet_content[idx])
        text_content = self.tokenizer.encode_plus(content, add_special_tokens = True, padding = 'max_length', truncation = True, max_length = self.max_sen_len, return_tensors = 'pt')
        
        dct_img = self.transform_dct(self.image[idx].convert('L'))
        dct_img = process_dct_img(dct_img)

        return {
            "text_input_ids": text_content["input_ids"].flatten().clone().detach().type(torch.LongTensor),
            "attention_mask": text_content["attention_mask"].flatten().clone().detach().type(torch.LongTensor),
            "token_type_ids": text_content["token_type_ids"].flatten().clone().detach().type(torch.LongTensor),
            "image": self.transform_vgg(self.image[idx]),
            "dct_img": dct_img,
            "post_id": self.post_id[idx],
            "label": self.label[idx],
        }
    def __len__(self):
        return len(self.label)