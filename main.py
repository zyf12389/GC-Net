import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable
import numpy as np
from read_data import sceneDisp
import torch.optim as optim

from gc_net import *
from python_pfm import *

#preprocess
def normalizeRGB(img):
    # for i in range(3):
    #     minval=torch.min(img[:,i, :, :])
    #     maxval=torch.max(img[:,i, :, :])
    #     if (minval.data!=maxval.data).cpu().numpy():
    #         img[:, i, :, :]=torch.div(img[:,i, :, :]-minval,maxval-minval)
    #         img[:,i,:,:]=torch.div(img[:,i, :, :]-0.5,0.5)
    # return img
    return img
tsfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

h=256
w=512
maxdisp=160 #gc_net.py also need to change  must be a multiple of 32...maybe can cancel the outpadding of deconv
batch=4
net = GcNet(h,w,maxdisp)
#net=net.cuda()
net=torch.nn.DataParallel(net).cuda()


#train
def train(epoch_total,loadstate):

    loss_mul_list = []
    for d in range(maxdisp):
        loss_mul_temp = Variable(torch.Tensor(np.ones([batch, 1, h, w]) * d)).cuda()
        loss_mul_list.append(loss_mul_temp)
    loss_mul = torch.cat(loss_mul_list, 1)

    optimizer = optim.RMSprop(net.parameters(), lr=0.001, alpha=0.9)
    dataset = sceneDisp('','train',tsfm)
    loss_fn=nn.L1Loss()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True,num_workers=1)

    imL = Variable(torch.FloatTensor(1).cuda())
    imR = Variable(torch.FloatTensor(1).cuda())
    dispL = Variable(torch.FloatTensor(1).cuda())

    loss_list=[]
    print(len(dataloader))
    start_epoch=0
    if loadstate==True:
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        accu=checkpoint['accur']
    print('startepoch:%d accuracy:%f' %(start_epoch,accu))
    for epoch in range(start_epoch,epoch_total):
        net.train()
        data_iter = iter(dataloader)

        print('\nEpoch: %d' % epoch)
        train_loss=0
        acc_total=0
        for step in range(len(dataloader)-1):
            print('----epoch:%d------step:%d------' %(epoch,step))
            data = next(data_iter)

            randomH = np.random.randint(0, 160)
            randomW = np.random.randint(0, 400)
            imageL = data['imL'][:,:,randomH:(randomH+h),randomW:(randomW+w)]
            imageR = data['imR'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
            disL = data['dispL'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
            imL.data.resize_(imageL.size()).copy_(imageL)
            imR.data.resize_(imageR.size()).copy_(imageR)
            dispL.data.resize_(disL.size()).copy_(disL)
            #normalize
            # imgL=normalizeRGB(imL)
            # imgR=normalizeRGB(imR)

            net.zero_grad()
            optimizer.zero_grad()

            x = net(imL,imR)
            # print(x.shape)
            # print(loss_mul.shape)
            # print(net)
            result=torch.sum(x.mul(loss_mul),1)
            # print(result.shape)
            tt=loss_fn(result,dispL)
            train_loss+=tt.data
            # tt = loss(x, loss_mul, dispL)
            tt.backward()
            optimizer.step()
            print('=======loss value for every step=======:%f' % (tt.data))
            print('=======average loss value for every step=======:%f' %(train_loss/(step+1)))
            result=result.view(batch,1,h,w)
            diff=torch.abs(result.data.cpu()-dispL.data.cpu())
            print(diff.shape)
            accuracy=torch.sum(diff<3)/float(h*w*batch)
            acc_total+=accuracy
            print('====accuracy for the result less than 3 pixels===:%f' %accuracy)
            print('====average accuracy for the result less than 3 pixels===:%f' % (acc_total/(step+1)))

            # save
            if step%100==0:
                loss_list.append(train_loss/(step+1))
            if (step>1 and step%200==0) or step==len(dataloader)-2:
                print('=======>saving model......')
                state={'net':net.state_dict(),'step':step,
                       'loss_list':loss_list,'epoch':epoch,'accur':acc_total}
                torch.save(state,'checkpoint/ckpt.t7')

                im = result[0, :, :, :].data.cpu().numpy().astype('uint8')
                im = np.transpose(im, (1, 2, 0))
                cv2.imwrite('train_result.png', im, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                gt = np.transpose(dispL[0, :, :, :].data.cpu().numpy(), (1, 2, 0))
                cv2.imwrite('train_gt.png', gt, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    fp=open('loss.txt','w')
    for i in range(len(loss_list)):
        fp.write(str(loss_list[i][0]))
        fp.write('\n')
    fp.close()


#test
def test(loadstate):

    if loadstate==True:
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        accu=checkpoint['accur']
    net.eval()
    imL = Variable(torch.FloatTensor(1).cuda())
    imR = Variable(torch.FloatTensor(1).cuda())
    dispL = Variable(torch.FloatTensor(1).cuda())

    dataset = sceneDisp('', 'test',tsfm)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    data_iter = iter(dataloader)
    data = next(data_iter)

    randomH = np.random.randint(0, 160)
    randomW = np.random.randint(0, 400)
    print('test')
    imageL = data['imL'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
    imageR = data['imR'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
    disL = data['dispL'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
    imL.data.resize_(imageL.size()).copy_(imageL)
    imR.data.resize_(imageR.size()).copy_(imageR)
    dispL.data.resize_(disL.size()).copy_(disL)
    loss_mul_list_test = []
    for d in range(maxdisp):
        loss_mul_temp = Variable(torch.Tensor(np.ones([1, 1, h, w]) * d)).cuda()
        loss_mul_list_test.append(loss_mul_temp)
    loss_mul_test = torch.cat(loss_mul_list_test, 1)

    with torch.no_grad():
        result=net(imL,imR)

    disp=torch.sum(result.mul(loss_mul_test),1)
    diff = torch.abs(disp.data.cpu() -dispL.data.cpu())  # end-point-error

    accuracy = torch.sum(diff < 3) / float(h * w)
    print('test accuracy less than 3 pixels:%f' %accuracy)

    # save
    im=disp.data.cpu().numpy().astype('uint8')
    im=np.transpose(im,(1,2,0))
    cv2.imwrite('test_result.png',im,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    gt=np.transpose(dispL[0,:,:,:].data.cpu().numpy(),(1,2,0))
    cv2.imwrite('test_gt.png',gt,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return disp

def main():
    epoch_total=20
    load_state=True
    train(epoch_total,load_state)
    test(load_state)


if __name__=='__main__':
    main()
