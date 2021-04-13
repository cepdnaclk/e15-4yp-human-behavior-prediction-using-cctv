from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import os
import scipy.io
from model import ft_net
import time

##############################################################################################
# Options
name = 'ft_ResNet50'
data_path = '../database/humans/2021-04-13'
model_path = '../model_data/reid/reid_model.pth'
output_path = '../database/humans/2021-04-13/today.mat'
stride = 2
batchsize = 32
nclasses = 751
gpu_id = 0

##############################################################################################
def extract_feature(model, dataloader):
    features = torch.FloatTensor()
    for data in dataloader:
        img, label = data
        #print(f'lable:{label}, type:{type(label)}')
        #print(type(img))
        #print(img.size())
        n, c, h, w = img.size()
        #print(f'n:{n}, c:{c}, h:{h}, w:{w}')
        #print(img.numpy()[7][0])
        ff = torch.FloatTensor(n,512).zero_().cuda()
        for i in range(2):
            if(i==1): #Flip the image
                inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
                img = img.index_select(3, inv_idx)

            input_img = Variable(img.cuda())
            outputs = model(input_img)
            ff += outputs

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features,ff.data.cpu()), 0)
    return features

def get_person_ids(image_paths):
    person_ids = []
    for path, v in image_paths:
        filename = os.path.basename(path)
        id = filename[0:4]
        person_ids.append(int(id))

    return person_ids

##############################################################################################

if __name__ == '__main__':

    # Set gpu ids
    torch.cuda.set_device(gpu_id)
    cudnn.benchmark = True

    # Load Data
    data_transform = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_dataset = datasets.ImageFolder(data_path, data_transform)
    #print(image_dataset)
    #print(type(image_dataset))
    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batchsize, shuffle=False, num_workers=0)
    #print(image_loader)
    #print(type(image_loader))

    model = ft_net(nclasses, stride = stride)
    model.load_state_dict(torch.load(model_path))
    model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    tic = time.time()
    # Extract feature
    with torch.no_grad():
        person_feature = extract_feature(model,image_loader)

    toc = time.time()
    print(f'ExtractTime: {toc - tic}s')

    image_paths = image_dataset.imgs
    person_ids = get_person_ids(image_paths)
    print(f'No Of Photos: {len(person_ids)}')
    print(f'Ids: {person_ids}')
    print(f'Feature Shape: {person_feature.numpy().shape}')
    #print(person_feature.numpy()[7])

    # Save to Matlab for check
    result = {'person_ids': person_ids, 'person_feature':person_feature.numpy()}
    scipy.io.savemat(output_path, result)