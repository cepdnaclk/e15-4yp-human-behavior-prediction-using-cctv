import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import numpy as np
import scipy.io
from torch.autograd import Variable

from BehaviorExtraction.reid.model import ft_net
# Options
name = 'ft_ResNet50'
stride = 2
batchsize = 256
nclasses = 751
score_threshold = 0.7
gpu_id = 0
name = ['Aunty', 'Risith', 'Asith', 'Madaya', 'YellowMan', 'BlueMan']

class HumanReIdentifier(object):
    def __init__(self, model_path, mat_path):
        # Set gpu ids
        torch.cuda.set_device(gpu_id)
        cudnn.benchmark = True

        result = scipy.io.loadmat(mat_path)
        self.human_feats = torch.FloatTensor(result['person_feature'])
        self.human_ids = result['person_ids'][0]

        self.reid_model = ft_net(nclasses, stride=stride)
        self.reid_model.load_state_dict(torch.load(model_path))
        self.reid_model.classifier.classifier = nn.Sequential()

        # Change to test mode
        self.reid_model = self.reid_model.eval()
        if torch.cuda.is_available():
            self.reid_model = self.reid_model.cuda()

    def extractFeatures(self, humanImg):
        feature = torch.FloatTensor()
        n, c, h, w = humanImg.size()
        ff = torch.FloatTensor(n, 512).zero_().cuda()
        for i in range(2):
            if (i == 1):  # Flip the image
                inv_idx = torch.arange(humanImg.size(3) - 1, -1, -1).long()
                humanImg = humanImg.index_select(3, inv_idx)

            input_img = Variable(humanImg.cuda())
            outputs = self.reid_model(input_img)
            ff += outputs

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        feature = torch.cat((feature, ff.data.cpu()), 0)
        return feature

    def getHumanTags(self, human_crops):
        stacked_tensor = torch.stack(human_crops)
        with torch.no_grad():
            feature = self.extractFeatures(stacked_tensor)

        query = torch.transpose(feature, 0, 1)
        score = torch.mm(self.human_feats, query)
        score = score.squeeze(1).cpu()

        tags = []
        indexes = np.argmax(score.numpy(), axis=0)

        if type(indexes) is not np.int64:
            scores = np.diag(score.numpy()[indexes])
        else:
            scores = [score.numpy()[indexes]]
            indexes = [indexes]

        for i, s in zip(indexes, scores):
            if s > score_threshold:
                tags.append(self.human_ids[i])
            else:
                tags.append(-1)

        return tags