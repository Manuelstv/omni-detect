# System libs
import torchvision
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
import re
import argparse
import sys

# Our libs
# import mit_semseg

import mit_semseg.models_sphe as seg_sphe
import mit_semseg.models as seg_persp

from mit_semseg.models import SegmentationModule
from mit_semseg.utils import colorEncode

#from torchviz import make_dot
#import hiddenlayer as hl
#from tensorflow.keras.metrics import MeanIoU

global layers_act 
layers_act = [False,False,False,False,False,False]

class OmniSemSeg():
    def __init__(self, datadir, savedir):

        self.colors = scipy.io.loadmat('data/color150.mat')['colors']

        # self.colors[2] = [255, 255, 255] #Sky
        # self.colors[4] = [25, 48, 16] #Trees
        # self.colors[13] = [0, 0, 0] #Ground

        self.colors[2] = [3,2,145] #Sky
        self.colors[4] = [0,214,0] #Trees
        self.colors[13] = [48,14,2] #Earth
        
        self.colors[91] = [48,14,2] #Ground
        self.colors[52] = [48,14,2] #Path
        # self.colors[16] = [48,14,2] #Mountain
        
        self.colors[17] = [0,214,0] #Plant
        # self.colors[106] = [0,214,0] #Canopy

        

        self.names = {}
        self.init_names()
        # for idx, elt in enumerate(self.colors):
        #     print(self.names[idx+1],self.colors[idx])

        self.model_sphe = self.model_builder("sphe")
        # x = torch.zeros([1,3,64,64])
        # y = self.model_sphe(x)
        # make_dot(y.mean(), params=dict(self.model_sphe.named_parameters()))
        # print(self.model_sphe)
        # self.model_sphe = self.model_builder("persp")
        self.model_persp = self.model_builder("persp")
        # print(self.model_persp)

        self.datadir = datadir
        self.ext = "_0.png"

        self.list_img = self.load_imgs()
        self.pil_to_tensor = self.img_transfrom()

        self.savedir = savedir


    def init_names(self):
        with open('data/object150_info.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.names[int(row[0])] = row[5].split(";")[0]

    def model_builder(self, imode="sphe"):
        encoder_epoch = 'ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth'
        decoder_epoch = 'ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth'
        # encoder_epoch = 'ckpt_nef/r50d_ppm_rot_e40_nef_30/encoder_epoch_40.pth'
        # decoder_epoch = 'ckpt_nef/r50d_ppm_rot_e40_nef_30/decoder_epoch_40.pth'
        if imode == "sphe":
            # Network Builders
            net_encoder = seg_sphe.ModelBuilder.build_encoder(
                arch='resnet50dilated',
                fc_dim=2048,
                weights=encoder_epoch)
            net_decoder = seg_sphe.ModelBuilder.build_decoder(
                arch='ppm_deepsup',
                fc_dim=2048,
                num_class=150,
                weights=decoder_epoch,
                use_softmax=True)
        elif imode == "persp":
            net_encoder = seg_persp.ModelBuilder.build_encoder(
                arch='resnet50dilated',
                fc_dim=2048,
                weights=encoder_epoch)
            net_decoder = seg_persp.ModelBuilder.build_decoder(
                arch='ppm_deepsup',
                fc_dim=2048,
                num_class=150,
                weights=decoder_epoch,
                use_softmax=True)

        crit = torch.nn.NLLLoss(ignore_index=-1)
        semseg_model = SegmentationModule(net_encoder, net_decoder, crit)
        semseg_model.eval()
        semseg_model.cuda()

        return semseg_model

    def img_transfrom(self):
        # Normalization parameters
         return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
                std=[0.229, 0.224, 0.225])  # across a large photo dataset.
        ])


    def load_imgs(self):
        # list of images to process
        # list_img = [self.datadir+file for file in sorted(os.listdir(self.datadir), key=lambda x:float(re.findall("(\d+)",x)[0])) if (file.endswith(self.ext))]
        list_img = sorted([self.datadir+file for file in os.listdir(self.datadir) if (file.endswith(self.ext))], key=lambda f: int(f.rsplit("/", 1)[-1].rsplit("_",1)[0]))
        # print(list_img)
        return list_img

    def batch_semseg_pred(self):
        for elt in self.list_img:
            self.semseg_pred(elt)

    def semseg_pred(self, elt):
        pil_image = PIL.Image.open(elt).convert('RGB')
        img_data = self.pil_to_tensor(pil_image)
        singleton_batch = {'img_data': img_data[None].cuda()}
        output_size = img_data.shape[1:]

        # Run the segmentation at the highest resolution.
        with torch.no_grad():
            scores_sphe = self.model_sphe(singleton_batch, segSize=output_size)
            # hl.build_graph(self.model_sphe, singleton_batch)
            # dot = make_dot(scores_sphe.mean(), params=dict(self.model_sphe.named_parameters()))
            # dot.format = 'png'
            # dot.render("net_semseg")
            # sys.exit()

        # Get the predicted scores for each pixel
        _, pred_sphe = torch.max(scores_sphe, dim=1)
        pred_sphe = pred_sphe.cpu()[0].numpy()

        # Run the segmentation at the highest resolution.
        with torch.no_grad():
            scores_persp = self.model_persp(singleton_batch, segSize=output_size)

        # Get the predicted scores for each pixel
        _, pred_persp = torch.max(scores_persp, dim=1)
        pred_persp = pred_persp.cpu()[0].numpy()

        return pred_sphe, pred_persp

    # def visualize_result(self, img, pred, index=None):

    #     pil_image = PIL.Image.open(img).convert('RGB')
    #     img_original = numpy.array(pil_image)

    #     # filter prediction class if requested
    #     if index is not None:
    #         pred = pred.copy()
    #         pred[pred != index] = -1
    #         print(f'{self.names[index+1]}:')

    #     # colorize prediction
    #     pred_color = colorEncode(pred, self.colors).astype(numpy.uint8)

    #     # aggregate images and save
    #     im_vis = numpy.concatenate((img_original, pred_color), axis=1)
    #     img_final = PIL.Image.fromarray(im_vis)

    # def save_result(self, img, pred, img_name, dir_result='./OUTPUT/', pre='', post=''):
    #     # colorize prediction
    #     pred_color = colorEncode(pred, self.colors).astype(numpy.uint8)

    #     # aggregate images and save
    #     im_vis = numpy.concatenate((img, pred_color), axis=1)
    #     img_final = PIL.Image.fromarray(im_vis)
    #     os.makedirs(dir_result, exist_ok=True)
    #     img_final.save(os.path.join(dir_result, pre+(img_name.split('/')[-1])[0:-4]+post+'.png'))

    def save_simple(self, img_orig, pred_persp, pred_sphe):
        # colorize prediction
        pred_persp_color = colorEncode(pred_persp, self.colors).astype(numpy.uint8)
        pred_sphe_color = colorEncode(pred_sphe, self.colors).astype(numpy.uint8)

        # aggregate images and save
        im_vis = numpy.concatenate((pred_persp_color, pred_sphe_color), axis=1)
        img_final = PIL.Image.fromarray(im_vis)

        new_im = PIL.Image.new('RGB', (img_final.size[0], 2*img_final.size[1]))

        new_im.paste(PIL.Image.open(img_orig))
        # it =  str(int((img_orig.split('/')[-1]).split('_')[0]))
        it =  str((img_orig.split('/')[-1]).split('_0')[0])
        gt_image = img_orig[0:-len((img_orig.split('/')[-1]))][0:-3]+'/2/'+it+'_2.png'
        # print(gt_image)
        # sys.exit()
        new_im.paste(PIL.Image.open(gt_image),(int(img_final.size[0]/2),0))
        new_im.paste(PIL.Image.fromarray(pred_persp_color),(0,img_final.size[1]))
        # new_im.paste(img_final,(0,img_final.size[1]))

        from PIL import ImageDraw, ImageFont

        img_edit = ImageDraw.Draw(new_im)
        text_color = (255, 255, 255)
        # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
        fnt = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 40)

        ipred_unique = numpy.unique(pred_persp[:,:], return_counts=True)[0]
        ipred_ratio =  10
        ipred_dist = int(img_final.size[1]/ipred_ratio)
        idx_loc = 0
        for ipred in ipred_unique:
            posx = int(img_final.size[0]*5/10) + 150 * numpy.floor(idx_loc/ipred_ratio)
            posy = img_final.size[1] + ipred_dist * (idx_loc%ipred_ratio) + ipred_dist/2
            img_edit.text((posx,posy), self.names[ipred+1], text_color, font=fnt, anchor="ls")
            img_edit.rectangle((posx-30,posy-20,posx-10,posy), fill=(self.colors[ipred][0],self.colors[ipred][1],self.colors[ipred][2]), outline=(255, 255, 255))
            idx_loc += 1


        os.makedirs(self.savedir, exist_ok=True)
        new_im.save(os.path.join(self.savedir, it+'.png'))
        

    def save_all(self, img_orig, pred_persp, pred_sphe):

        # pil_image = PIL.Image.open(img_orig).convert('RGB')
        # img_original = numpy.array(pil_image)

        # colorize prediction
        pred_persp_color = colorEncode(pred_persp, self.colors).astype(numpy.uint8)
        # pred_persp_color = (pred_persp).astype(numpy.uint8)
        pred_sphe_color = colorEncode(pred_sphe, self.colors).astype(numpy.uint8)

        # aggregate images and save
        im_vis = numpy.concatenate((pred_persp_color, pred_sphe_color), axis=1)
        img_final = PIL.Image.fromarray(im_vis)
        #print(img_final.size)

        new_im = PIL.Image.new('RGB', (img_final.size[0], 2*img_final.size[1]))

        new_im.paste(PIL.Image.open(img_orig))
        it =  str(int((img_orig.split('/')[-1]).split('_')[0]))
        gt_image = img_orig[0:-len((img_orig.split('/')[-1]))]+it+'_2.png'
        #print(gt_image)
        new_im.paste(PIL.Image.open(gt_image),(int(img_final.size[0]/2),0))
        new_im.paste(img_final,(0,img_final.size[1]))

        os.makedirs(self.savedir, exist_ok=True)
        # print(it)
        # print(img_orig)
        new_im.save(os.path.join(self.savedir, it+'.png'))

        # numpy.savetxt(os.path.join(self.savedir, it+'_sphe.csv'),pred_sphe, delimiter=',')
        # numpy.savetxt(os.path.join(self.savedir, it+'_persp.csv'),pred_persp, delimiter=',')
        # numpy.save(os.path.join(self.savedir, it+'_sphe.npy'),pred_sphe)
        # numpy.save(os.path.join(self.savedir, it+'_persp.npy'),pred_persp)

    def save_all_2(self, img_orig, pred_persp, pred_sphe):

        # pil_image = PIL.Image.open(img_orig).convert('RGB')
        # img_original = numpy.array(pil_image)

        # colorize prediction
        pred_persp_color = colorEncode(pred_persp, self.colors).astype(numpy.uint8)
        # pred_persp_color = (pred_persp).astype(numpy.uint8)
        pred_sphe_color = colorEncode(pred_sphe, self.colors).astype(numpy.uint8)

        # aggregate images and save
        im_vis = numpy.concatenate((pred_persp_color, pred_sphe_color), axis=1)
        img_final = PIL.Image.fromarray(im_vis)
        #print(img_final.size)

        new_im = PIL.Image.new('RGB', (img_final.size[0], 2*img_final.size[1]))

        new_im.paste(PIL.Image.open(img_orig))
        it =  str(int((img_orig.split('/')[-1]).split('_')[0]))
        gt_image = img_orig[0:-len((img_orig.split('/')[-1]))]+it+'_2.png'
        #print(gt_image)
        new_im.paste(PIL.Image.open(gt_image),(int(img_final.size[0]/2),0))
        new_im.paste(img_final,(0,img_final.size[1]))



        from PIL import ImageDraw, ImageFont

        img_edit = ImageDraw.Draw(new_im)
        text_color = (255, 255, 255)
        # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
        fnt = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 40)


        ipred_unique = numpy.unique(pred_persp[:,:], return_counts=True)[0]
        ipred_ratio =  10
        ipred_dist = int(img_final.size[1]/ipred_ratio)
        idx_loc = 0
        for ipred in ipred_unique:
            # print(ipred+1)
            posx = int(img_final.size[0]*4/10) + 150 * numpy.floor(idx_loc/ipred_ratio)
            posy = img_final.size[1] + ipred_dist * (idx_loc%ipred_ratio) + ipred_dist/2
            # print(off_text,ipred_dist)
            # print(idx_loc%ipred_ratio)
            # print(numpy.floor(idx_loc/ipred_ratio))
            # if posy >= img_final.size[1]*2:
            #     posx = int(img_final.size[0]*4/10) + 100 * numpy.floor(off_text/ipred_dist)
            #     posy = img_final.size[1]+(off_text%ipred_dist)
            img_edit.text((posx,posy), self.names[ipred+1], text_color, font=fnt, anchor="ls")
            img_edit.rectangle((posx-30,posy-20,posx-10,posy), fill=(self.colors[ipred][0],self.colors[ipred][1],self.colors[ipred][2]), outline=(255, 255, 255))
            idx_loc += 1

        ipred_unique = numpy.unique(pred_sphe[:,:], return_counts=True)[0]
        idx_loc = 0
        for ipred in ipred_unique:
            # print(ipred+1)
            posx = int(img_final.size[0]*9/10) + 150 * numpy.floor(idx_loc/ipred_ratio)
            posy = img_final.size[1] + ipred_dist * (idx_loc%ipred_ratio) + ipred_dist/2
            img_edit.text((posx,posy), self.names[ipred+1], text_color, font=fnt, anchor="ls")
            img_edit.rectangle((posx-30,posy-20,posx-10,posy), fill=(self.colors[ipred][0],self.colors[ipred][1],self.colors[ipred][2]), outline=(255, 255, 255))
            idx_loc += 1

        os.makedirs(self.savedir, exist_ok=True)
        # print(it)
        # print(img_orig)
        new_im.save(os.path.join(self.savedir, it+'.png'))


    # def merge_imgs(dir_result='/MERGED/'):
    #     dir_name = './OUTPUT/'
    #     os.makedirs(dir_name+dir_result, exist_ok=True)
    #     list_folders = ['ALL_OFFSETS','DECODER_NO_OFFSETS','BOTTLENECK_OFFSETS','123_LAYER_OFFSETS','FIRST_LAYER_OFFSETS','ENCODER_NO_OFFSETS','NO_OFFSETS']
    #     list_img_off = [dir_name+list_folders[0]+'/'+file for file in sorted(os.listdir(dir_name+list_folders[0])) if file.endswith('.png')]
    #     #print(len(list_img_off))
    #     for idx in range(len(list_img_off)):
    #         first_img = PIL.Image.open(dir_name+list_folders[0]+'/'+str(idx)+'.png')

    #         new_im = PIL.Image.new('RGB', (first_img.size[0], int((len(list_folders)+1)/2*first_img.size[1])))

    #         for k in range(len(list_folders)):
    #             #print(dir_name+list_folders[k]+'/'+str(idx)+'.png')
    #             new_im.paste(PIL.Image.open(dir_name+list_folders[k]+'/'+str(idx)+'.png'),(0,int((len(list_folders)-k-1)/2*first_img.size[1])))

    #         im_draw = PIL.ImageDraw.Draw(new_im)
    #         text_color = (0, 0, 0)
    #         #fnt = PIL.ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
    #         fnt = PIL.ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 40)
    #         for k in range(len(list_folders)):
    #             im_draw.text((int(first_img.size[0]/2),int((len(list_folders)-k)/2*first_img.size[1]+50)), str(list_folders[k]), text_color, font=fnt, anchor="ms")

    #         new_im.save(os.path.join(dir_name+dir_result, str(idx)+'.png'))
    #     print('FINI')



    # def merge_imgs_v2(dir_result='/MERGED_2/'):
    #     dir_name = './OUTPUT/'
    #     os.makedirs(dir_name+dir_result, exist_ok=True)
    #     list_folders = ['ALL_OFFSETS','DECODER_NO_OFFSETS','BOTTLENECK_OFFSETS','123_LAYER_OFFSETS','FIRST_LAYER_OFFSETS','ENCODER_NO_OFFSETS','NO_OFFSETS']
    #     list_img_off = [dir_name+list_folders[0]+'/'+file for file in sorted(os.listdir(dir_name+list_folders[0])) if file.endswith('.png')]
    #     #print(len(list_img_off))
    #     for idx in range(len(list_img_off)):
    #         first_img = PIL.Image.open(dir_name+list_folders[0]+'/'+str(idx)+'.png')

    #         new_im = PIL.Image.new('RGB', (int(first_img.size[0]/2), int((len(list_folders)+1)/2*first_img.size[1])))

    #         for k in range(len(list_folders)-1):
    #             #print(dir_name+list_folders[k]+'/'+str(idx)+'.png')
    #             new_im.paste(PIL.Image.open(dir_name+list_folders[k]+'/'+str(idx)+'.png'),(-int(first_img.size[0]/2),int((len(list_folders)-k-1)/2*first_img.size[1])))

    #         new_im.paste(PIL.Image.open(dir_name+list_folders[k]+'/'+str(idx)+'.png'),(0,0))

    #         im_draw = PIL.ImageDraw.Draw(new_im)
    #         text_color = (0, 0, 0)
    #         #fnt = PIL.ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
    #         fnt = PIL.ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 40)
    #         for k in range(len(list_folders)):
    #             im_draw.text((int(first_img.size[0]/4),int((len(list_folders)-k)/2*first_img.size[1]+50)), str(list_folders[k]), text_color, font=fnt, anchor="ms")

    #         new_im.save(os.path.join(dir_name+dir_result, str(idx)+'.png'))
    #     print('FINI')



# def accuracy(preds, label):
#     valid = (label >= 0)
#     acc_sum = (valid * (preds == label)).sum()
#     valid_sum = valid.sum()
#     acc = float(acc_sum) / (valid_sum + 1e-10)
#     return acc, valid_sum

# def intersectionAndUnion(imPred, imLab, numClass):
#     imPred = np.asarray(imPred).copy()
#     imLab = np.asarray(imLab).copy()

#     imPred += 1
#     imLab += 1
#     # Remove classes from unlabeled pixels in gt image.
#     # We should not penalize detections in unlabeled portions of the image.
#     imPred = imPred * (imLab > 0)

#     # Compute area intersection:
#     intersection = imPred * (imPred == imLab)
#     (area_intersection, _) = np.histogram(
#         intersection, bins=numClass, range=(1, numClass))

#     # Compute area union:
#     (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
#     (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
#     area_union = area_pred + area_lab - area_intersection

#     return (area_intersection, area_union)





# def show_comparison_pred():
#     # Load and normalize one image as a singleton tensor batch
#     pil_to_tensor = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
#             std=[0.229, 0.224, 0.225])  # across a large photo dataset.
#     ])
#     pil_image = PIL.Image.open('/media/cartizzu/DATA/DATASETS/RICOH/ZOE/ZOE_5/out-632.png').convert('RGB')
#     img_original = numpy.array(pil_image)
#     img_data = pil_to_tensor(pil_image)
#     singleton_batch = {'img_data': img_data[None].cuda()}
#     output_size = img_data.shape[1:]
#     # Run the segmentation at the highest resolution.
#     with torch.no_grad():
#         scores = segmentation_module_sphe(singleton_batch, segSize=output_size)

#     # Get the predicted scores for each pixel
#     _, pred = torch.max(scores, dim=1)
#     pred = pred.cpu()[0].numpy()
#     visualize_result(img_original, pred)
#     # Run the segmentation at the highest resolution.

#     with torch.no_grad():
#         scores = segmentation_module_persp(singleton_batch, segSize=output_size)

#     # Get the predicted scores for each pixel
#     _, pred = torch.max(scores, dim=1)
#     pred = pred.cpu()[0].numpy()
#     visualize_result(img_original, pred)


PARSER = argparse.ArgumentParser()
PARSER.add_argument('-d', '--datadir',
                    nargs='?',
                    type=str,
                    default='/media/cartizzu/DATA/DATASETS/UNREAL/FOREST/FOREST_30/',
                    help='Source directory containing the cubemap images in a '
                         '\'CUBEMAP\' folder (defaults to ./CAPTURES).')
PARSER.add_argument('-s', '--savedir',
                    nargs='?',
                    type=str,
                    default='./OUTPUT/',
                    help='Source directory containing the cubemap images in a '
                         '\'CUBEMAP\' folder (defaults to ./CAPTURES).')
PARSER.add_argument('-m', '--mode',
                    nargs='?',
                    type=str,
                    default='test',
                    help='Mode of execution test or eval vs GT.')
PARSER.add_argument('-v', '--VERBOSE',
                    nargs='*',
                    action='store',
                    help='If true, prints out additional info.')


from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
import time

class semseg_metric():
    def __init__(self):
        self.acc_meter = AverageMeter()
        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.time_meter = AverageMeter()

    def update_metrics(self, pred_color, semseg_gt, timetic):
        self.time_meter.update(timetic)
        # calculate accuracy
        acc, pix = accuracy(pred_color, semseg_gt)
        intersection, union = intersectionAndUnion(pred_color, semseg_gt, 150) # 150 nb of class in dataset
        self.acc_meter.update(acc, pix)
        self.intersection_meter.update(intersection)
        self.union_meter.update(union)

    def show_metrics(self, imode):
        print("Metric for ",str(imode))
        iou = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
        # for i, _iou in enumerate(iou):
        #     print('class [{}], IoU: {:.8f}'.format(i, _iou))

        print('[Eval Summary]:')
        print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
              .format(iou.mean(), self.acc_meter.average()*100, self.time_meter.average()))
        print()




def iou_mean(pred, target, n_classes = 1):
    #n_classes ：the number of classes in your dataset,not including background
    # for mask and ground-truth label, not probability map
      ious = []
      iousSum = 0
      pred = torch.from_numpy(pred)
      pred = pred.view(-1)
      target = numpy.array(target)
      target = torch.from_numpy(target)
      target = target.view(-1)

      # Ignore IoU for background class ("0")
      for cls in range(1, n_classes+1):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
          ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
          ious.append(float(intersection) / float(max(union, 1)))
          iousSum += float(intersection) / float(max(union, 1))
      return iousSum/n_classes



def main():
    """Run main function"""

    # layers_act = [True,True,True,True,True]

    OSS = OmniSemSeg(DATADIR, SAVEDIR)

    if VERBOSE:
        print('Semantic Segmentation ')

        print("Saving results to %s" % SAVEDIR)

    print("Nombre images: ",len(OSS.list_img))
    
    if IMODE == "test":
        for elt in OSS.list_img:
            torch.cuda.synchronize()
            tic = time.perf_counter()

            pred_sphe, pred_persp = OSS.semseg_pred(elt)

            time_end = time.perf_counter() - tic
            # if VERBOSE:
            print("Done for ",str(elt), "in ", time_end)
            OSS.save_simple(elt, pred_persp, pred_sphe)
            # OSS.save_all(elt, pred_persp, pred_sphe)

    elif IMODE == "eval":


        from mit_semseg.lib.utils import as_numpy

        semseg_metric_persp = semseg_metric()
        semseg_metric_sphe = semseg_metric()

        for elt in OSS.list_img:


            semseg_gt_file = elt.replace("_0.png","_2.png")
            semseg_gt = as_numpy(PIL.Image.open(semseg_gt_file).convert('RGB'))
            # print("Image seg GT")
            # # print(semseg_gt)
            # print(numpy.unique(semseg_gt[:,:,0], return_counts=True)) #red
            # print(numpy.unique(semseg_gt[:,:,1], return_counts=True)) #green
            # print(numpy.unique(semseg_gt[:,:,2], return_counts=True)) #blue
            # semseg_gt_id = numpy.zeros((semseg_gt.shape[0],semseg_gt.shape[1])) -1
            # for idx in range(semseg_gt.shape[0]):
            #     for idy in range(semseg_gt.shape[1]):
            #         for idc, col in enumerate(OSS.colors):
            #             if not((semseg_gt[idx,idy] - col).all()):
            #                 semseg_gt_id[idx,idy] = idc
            #                 break
            # print("Semseg Gt ID")
            # print(semseg_gt_id)

            torch.cuda.synchronize()
            tic = time.perf_counter()

            # if VERBOSE:
            print("Doing for ",str(elt))
            pred_sphe, pred_persp = OSS.semseg_pred(elt)
            # OSS.save_all(elt, pred_persp, pred_sphe)
            OSS.save_all_2(elt, pred_persp, pred_sphe)

            pred_sphe_color = colorEncode(pred_sphe, OSS.colors).astype(numpy.uint8)
            pred_persp_color = colorEncode(pred_persp, OSS.colors).astype(numpy.uint8)

            semseg_metric_persp.update_metrics(pred_persp_color,semseg_gt,time.perf_counter() - tic)
            semseg_metric_sphe.update_metrics(pred_sphe_color,semseg_gt,time.perf_counter() - tic)



            # print("MIOU KERAS : ",iou_mean(pred_sphe,semseg_gt_id,150))

        semseg_metric_persp.show_metrics("PERSP")
        semseg_metric_sphe.show_metrics("SPHE")


    print("DONE")

if __name__ == '__main__':
    args = PARSER.parse_args()
    DATADIR = args.datadir
    SAVEDIR = args.savedir
    IMODE = args.mode
    VERBOSE = args.VERBOSE is not None
    main()
    

