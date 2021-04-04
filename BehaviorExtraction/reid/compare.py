from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import os
import scipy.io
from model import ft_net
import cv2
import numpy as np
import time
import glob
##############################################################################################
# Options
name = 'ft_ResNet50'
data_path = '../database/2021-02-05'
model_path = '../model_data/ft_ResNet50_ReID/net_last.pth'
mat_path = '../database/2021-02-05/features.mat'
stride = 2
batchsize = 256
nclasses = 751
score_threshold = 0.3
gpu_id = 0

##############################################################################################
def extract_feature(model, img):
    feature = torch.FloatTensor()
    n, c, h, w = img.size()
    print(f'n:{n}, c:{c}, h:{h}, w:{w}')

    ff = torch.FloatTensor(n, 512).zero_().cuda()
    for i in range(2):
        if (i == 1):  # Flip the image
            inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
            img = img.index_select(3, inv_idx)

        input_img = Variable(img.cuda())
        outputs = model(input_img)
        ff += outputs

    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    feature = torch.cat((feature, ff.data.cpu()), 0)
    return feature

def get_image_tensor():
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_tensor_list = []
    paths = []
    for filename in os.listdir("../database/test"):
        path = os.path.join("../database/test", filename)
        image = cv2.imread(path)
        if image is not None:
            paths.append(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor_list.append(data_transform(image))

    stacked_tensor = torch.stack(image_tensor_list)
    return stacked_tensor, paths

##############################################################################################
if __name__ == '__main__':
    tic1 = time.time()
    # Set gpu ids
    torch.cuda.set_device(gpu_id)
    cudnn.benchmark = True

    result = scipy.io.loadmat(mat_path)
    person_feature = torch.FloatTensor(result['person_feature'])
    person_ids = result['person_ids'][0]

    model = ft_net(nclasses, stride=stride)
    model.load_state_dict(torch.load(model_path))
    model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if torch.cuda.is_available():
        person_features = person_feature.cuda()
        model = model.cuda()

    images_tensor, paths = get_image_tensor()
    tic2 = time.time()
    # Extract feature
    with torch.no_grad():
        feature = extract_feature(model, images_tensor)

    toc = time.time()
    print(f'OverallTime: {toc - tic1}')
    print(f'ExtractTime: {toc - tic2}')

    query = torch.transpose(feature, 0, 1)
    score = torch.mm(person_feature, query)
    score = score.squeeze(1).cpu()

    indexes = np.argmax(score.numpy(), axis=0)
    scores = np.diag(score.numpy()[indexes])

    name = ['Aunty', 'Risith', 'Asith', 'Madaya', 'YellowMan', 'BlueMan']
    for p, i, s in zip(paths, indexes, scores):
        if s > score_threshold:
            print(f'{p} -> ({s}) {}')
        else:
            print(f'{p} -> ({s}) Unknown')


