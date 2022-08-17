# System libs
import time
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
import torchvision
import os
import csv
import torch
import numpy
import scipy.io
import PIL.Image
import torchvision.transforms
import re
import argparse
import sys
import numpy as np
import tensorflow
from mit_semseg.lib.utils import as_numpy
import mit_semseg.utils_v2 as u_v2

from PIL import ImageDraw, ImageFont

# Our libs
# import mit_semseg

import mit_semseg.models_v2 as seg_sphe
import mit_semseg.models as seg_persp

from mit_semseg.models import SegmentationModule
from mit_semseg.utils import colorEncode

#from torchviz import make_dot
#import hiddenlayer as hl
#from tensorflow.keras.metrics import MeanIoU

global layers_act
layers_act = [False, False, False, False, False, False]

ctrees = [11, 236, 9]
cground = [153, 108, 6]
csky = [29, 26, 199]

id_trees = 4
id_earth = 13
id_sky = 2

id_ground = 94
id_path = 52
id_mountain = 16
id_dirt = 91
id_hill = 68

id_plants = 17
id_canopy = 106

nb_classes_in_dataset = 150
model_version = "semseg_baseline_fusedclasses_RICOH"
top_half = False

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-d', '--datadir',
                    nargs='?',
                    type=str,
                    default='/media/cartizzu/DATA/LIN/2_CODE/4_SEGMENTATION/semantic-segmentation-pytorch/OUTPUT/TEST/',
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
PARSER.add_argument('-th', '--top_half',
                    nargs='?',
                    type=bool,
                    default=False,
                    help='Remove top bottom of images for metrics.')
PARSER.add_argument('-v', '--VERBOSE',
                    nargs='*',
                    action='store',
                    help='If true, prints out additional info.')


def init_metrics(nb_classes_in_dataset):
    gmiou = tensorflow.keras.metrics.MeanIoU(num_classes=nb_classes_in_dataset)
    giou_trees = tensorflow.keras.metrics.IoU(num_classes=nb_classes_in_dataset, target_class_ids=[id_trees])
    giou_ground = tensorflow.keras.metrics.IoU(num_classes=nb_classes_in_dataset, target_class_ids=[id_earth])
    giou_sky = tensorflow.keras.metrics.IoU(num_classes=nb_classes_in_dataset, target_class_ids=[id_sky])
    miou = tensorflow.keras.metrics.MeanIoU(num_classes=nb_classes_in_dataset)
    iou_trees = tensorflow.keras.metrics.IoU(num_classes=nb_classes_in_dataset, target_class_ids=[id_trees])
    iou_ground = tensorflow.keras.metrics.IoU(num_classes=nb_classes_in_dataset, target_class_ids=[id_earth])
    iou_sky = tensorflow.keras.metrics.IoU(num_classes=nb_classes_in_dataset, target_class_ids=[id_sky])

    gmiou.reset_state()
    giou_trees.reset_state()
    giou_ground.reset_state()
    giou_sky.reset_state()

    acc_list = np.array([])
    giou_list = np.empty(4)
    iou_list = np.empty(4)

    iou_vector = [gmiou, giou_trees, giou_ground, giou_sky, miou, iou_trees, iou_ground, iou_sky]

    return iou_vector, acc_list, giou_list, iou_list


def update_metrics(semseg_gt_id, seg_pred, iou_vector, acc_list, giou_list, iou_list, top_half=False):
    [gmiou, giou_trees, giou_ground, giou_sky, miou, iou_trees, iou_ground, iou_sky] = iou_vector
    # print(seg_pred[-1,0])
    # if top_half:
    #     print("CAREFUL TOP HALF REDUCTION IS ACTIV!!")
    #     half_height = int(seg_pred.shape[0] / 2)
    #     # print(half_height)
    #     seg_pred = seg_pred[:half_height, :]
    #     semseg_gt_id = semseg_gt_id[:half_height, :]
    gmiou.update_state(semseg_gt_id, seg_pred)
    giou_trees.update_state(semseg_gt_id, seg_pred)
    giou_ground.update_state(semseg_gt_id, seg_pred)
    giou_sky.update_state(semseg_gt_id, seg_pred)

    miou.reset_state()
    iou_trees.reset_state()
    iou_ground.reset_state()
    iou_sky.reset_state()
    miou.update_state(semseg_gt_id, seg_pred)
    iou_trees.update_state(semseg_gt_id, seg_pred)
    iou_ground.update_state(semseg_gt_id, seg_pred)
    iou_sky.update_state(semseg_gt_id, seg_pred)

    acc = np.mean((semseg_gt_id == seg_pred))
    acc_list = np.append(acc_list, acc)

    tmp_giou = [gmiou.result().numpy(), giou_trees.result().numpy(), giou_ground.result().numpy(), giou_sky.result().numpy()]
    tmp_iou = [miou.result().numpy(), iou_trees.result().numpy(), iou_ground.result().numpy(), iou_sky.result().numpy()]

    giou_list = np.vstack((giou_list, tmp_giou))
    iou_list = np.vstack((iou_list, tmp_iou))

    iou_vector = [gmiou, giou_trees, giou_ground, giou_sky, miou, iou_trees, iou_ground, iou_sky]

    return iou_vector, acc_list, giou_list, iou_list


class OmniSemSeg():
    def __init__(self, datadir, savedir):

        self.colors = scipy.io.loadmat('data/color150.mat')['colors']

        # self.colors[2] = [255, 255, 255] #Sky
        # self.colors[4] = [25, 48, 16] #Trees
        # self.colors[13] = [0, 0, 0] #Ground

        self.colors[id_trees] = ctrees  # Trees
        self.colors[id_earth] = cground  # Earth
        self.colors[id_sky] = csky  # Sky

        self.names = {}
        self.init_names()
        # for idx, elt in enumerate(self.colors):
        #     print(self.names[idx+1],self.colors[idx])

        self.model_sphe = self.model_builder("sphe")
        print(self.model_sphe)

        self.model_persp = self.model_builder("persp")
        # print(self.model_persp)

        # self.datadir = os.path.join(datadir, "INPUT/")
        # self.datadir = os.path.join(datadir, "INPUT_SSHORT/")
        self.datadir = os.path.join(datadir, "images/")
        self.ext = "_rgb.png"

        self.list_img = self.load_imgs()
        self.pil_to_tensor = self.img_transfrom()

        self.savedir = datadir

        self.ipred_ratio = 20
        self.fnt = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 15)

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
                arch='resnet50dilated_sphe',
                fc_dim=2048,
                weights=encoder_epoch)
            net_decoder = seg_sphe.ModelBuilder.build_decoder(
                arch='ppm_deepsup_sphe',
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
                mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
                std=[0.229, 0.224, 0.225])  # across a large photo dataset.
        ])

    def load_imgs(self):
        # list of images to process
        # list_img = sorted([self.datadir+file for file in os.listdir(self.datadir) if (file.endswith(self.ext))], key=lambda f: int(f.rsplit("/", 1)[-1].rsplit("_", 1)[0]))
        list_img = sorted([self.datadir+f for f in sorted(os.listdir(self.datadir)) if (os.path.isfile(os.path.join(self.datadir, f)) and f.endswith(self.ext))])
        # print(list_img)
        return list_img

    def batch_semseg_pred(self):
        for elt in self.list_img:
            self.semseg_pred(elt)

    def semseg_single_pred(self, elt):
        pil_image = PIL.Image.open(elt).convert('RGB')
        img_data = self.pil_to_tensor(pil_image)
        singleton_batch = {'img_data': img_data[None].cuda()}
        output_size = img_data.shape[1:]

        # Run the segmentation at the highest resolution.
        with torch.no_grad():
            scores_seg = self.model_persp(singleton_batch, segSize=output_size)
            # scores_seg = self.model_sphe(singleton_batch, segSize=output_size)

        # Get the predicted scores for each pixel
        _, pred_seg = torch.max(scores_seg, dim=1)
        pred_seg = pred_seg.cpu()[0].numpy()

        # To fuse classes of objects
        pred_seg[pred_seg == id_path] = id_earth
        pred_seg[pred_seg == id_dirt] = id_earth
        pred_seg[pred_seg == id_mountain] = id_earth
        pred_seg[pred_seg == id_ground] = id_earth
        pred_seg[pred_seg == id_hill] = id_earth

        pred_seg[pred_seg == id_plants] = id_trees
        pred_seg[pred_seg == id_canopy] = id_trees

        return pred_seg

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

    # def save_simple(self, img_orig, pred_persp, pred_sphe):
    #     name_img = (img_orig.split('/')[-1])[0:-8]
    #     # colorize prediction
    #     pred_persp_color = colorEncode(pred_persp, self.colors).astype(numpy.uint8)
    #     pred_sphe_color = colorEncode(pred_sphe, self.colors).astype(numpy.uint8)

    #     # aggregate images and save
    #     im_vis = numpy.concatenate((pred_persp_color, pred_sphe_color), axis=1)
    #     img_final = PIL.Image.fromarray(im_vis)

    #     new_im = PIL.Image.new('RGB', (img_final.size[0], 2*img_final.size[1]))

    #     new_im.paste(PIL.Image.open(img_orig))
    #     gt_image = img_orig.replace("_rgb.png", "_seg.png")
    #     new_im.paste(PIL.Image.open(gt_image), (int(img_final.size[0]/2), 0))
    #     new_im.paste(PIL.Image.fromarray(pred_persp_color), (0, img_final.size[1]))

    #     from PIL import ImageDraw, ImageFont

    #     img_edit = ImageDraw.Draw(new_im)
    #     text_color = (255, 255, 255)
    #     # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
    #     fnt = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 40)

    #     ipred_unique = numpy.unique(pred_persp[:, :], return_counts=True)[0]
    #     ipred_dist = int(img_final.size[1]/self.ipred_ratio)
    #     idx_loc = 0
    #     for ipred in ipred_unique:
    #         posx = int(img_final.size[0]*5/10) + 150 * numpy.floor(idx_loc/self.ipred_ratio)
    #         posy = img_final.size[1] + ipred_dist * (idx_loc % self.ipred_ratio) + ipred_dist/2
    #         img_edit.text((posx, posy), self.names[ipred+1], text_color, font=fnt, anchor="ls")
    #         img_edit.rectangle((posx-30, posy-20, posx-10, posy), fill=(self.colors[ipred][0], self.colors[ipred][1], self.colors[ipred][2]), outline=(255, 255, 255))
    #         idx_loc += 1

    #     os.makedirs(self.savedir, exist_ok=True)
    #     new_im.save(os.path.join(self.savedir,  name_img + '_pred.png'))

    # def save_all(self, img_orig, pred_persp, pred_sphe):

    #     # pil_image = PIL.Image.open(img_orig).convert('RGB')
    #     # img_original = numpy.array(pil_image)

    #     # colorize prediction
    #     pred_persp_color = colorEncode(pred_persp, self.colors).astype(numpy.uint8)
    #     # pred_persp_color = (pred_persp).astype(numpy.uint8)
    #     pred_sphe_color = colorEncode(pred_sphe, self.colors).astype(numpy.uint8)

    #     # aggregate images and save
    #     im_vis = numpy.concatenate((pred_persp_color, pred_sphe_color), axis=1)
    #     img_final = PIL.Image.fromarray(im_vis)
    #     #print(img_final.size)

    #     new_im = PIL.Image.new('RGB', (img_final.size[0], 2*img_final.size[1]))

    #     new_im.paste(PIL.Image.open(img_orig))
    #     it = str(int((img_orig.split('/')[-1]).split('_')[0]))
    #     gt_image = img_orig[0:-len((img_orig.split('/')[-1]))]+it+'_2.png'
    #     #print(gt_image)
    #     new_im.paste(PIL.Image.open(gt_image), (int(img_final.size[0]/2), 0))
    #     new_im.paste(img_final, (0, img_final.size[1]))

    #     os.makedirs(self.savedir, exist_ok=True)
    #     # print(it)
    #     # print(img_orig)
    #     new_im.save(os.path.join(self.savedir, it+'.png'))

    #     # numpy.savetxt(os.path.join(self.savedir, it+'_sphe.csv'),pred_sphe, delimiter=',')
    #     # numpy.savetxt(os.path.join(self.savedir, it+'_persp.csv'),pred_persp, delimiter=',')
    #     # numpy.save(os.path.join(self.savedir, it+'_sphe.npy'),pred_sphe)
    #     # numpy.save(os.path.join(self.savedir, it+'_persp.npy'),pred_persp)

    def save_single_nogt(self, img_orig, pred_seg, model_version):
        name_img = (img_orig.split('/')[-1])[0:-8]

        # colorize prediction
        pred_seg_color = colorEncode(pred_seg, self.colors).astype(numpy.uint8)

        # aggregate images and save
        img_final = PIL.Image.fromarray(pred_seg_color)

        new_im = PIL.Image.new('RGB', (img_final.size[0], 2*img_final.size[1]))

        new_im.paste(PIL.Image.open(img_orig))
        # gt_image = os.path.join(self.datadir, name_img+'_seg.png')
        # new_im.paste(PIL.Image.open(gt_image), (int(img_final.size[0]/2), 0))
        new_im.paste(img_final, (0, img_final.size[1]))

        img_edit = ImageDraw.Draw(new_im)
        text_color = (255, 255, 255)

        ipred_unique = numpy.unique(pred_seg[:, :], return_counts=True)[0]
        ipred_ratio = 10
        ipred_dist = int(img_final.size[1]/ipred_ratio)
        idx_loc = 0
        for ipred in ipred_unique:
            # print(ipred+1)
            posx = int(img_final.size[0]*4/10) + 150 * numpy.floor(idx_loc/ipred_ratio)
            posy = img_final.size[1] + ipred_dist * (idx_loc % ipred_ratio) + ipred_dist/2
            img_edit.text((posx, posy), self.names[ipred+1], text_color, font=self.fnt, anchor="ls")
            img_edit.rectangle((posx-30, posy-20, posx-10, posy), fill=(self.colors[ipred][0], self.colors[ipred][1], self.colors[ipred][2]), outline=(255, 255, 255))
            idx_loc += 1

        save_dir = os.path.join(self.savedir, model_version)
        os.makedirs(save_dir, exist_ok=True)
        new_im.save(os.path.join(save_dir,  name_img + '_pred.png'))

    def save_all_2(self, img_orig, pred_persp, pred_sphe, model_version):
        name_img = (img_orig.split('/')[-1])[0:-8]

        # colorize prediction
        pred_persp_color = colorEncode(pred_persp, self.colors).astype(numpy.uint8)
        # pred_persp_color = (pred_persp).astype(numpy.uint8)
        pred_sphe_color = colorEncode(pred_sphe, self.colors).astype(numpy.uint8)

        # aggregate images and save
        im_vis = numpy.concatenate((pred_persp_color, pred_sphe_color), axis=1)
        img_final = PIL.Image.fromarray(im_vis)

        new_im = PIL.Image.new('RGB', (img_final.size[0], 2*img_final.size[1]))

        new_im.paste(PIL.Image.open(img_orig))
        gt_image = img_orig.replace("_rgb.png", "_seg.png")
        new_im.paste(PIL.Image.open(gt_image), (int(img_final.size[0]/2), 0))
        new_im.paste(img_final, (0, img_final.size[1]))

        img_edit = ImageDraw.Draw(new_im)
        text_color = (255, 255, 255)
        # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)

        ipred_unique = numpy.unique(pred_persp[:, :], return_counts=True)[0]

        ipred_dist = int(img_final.size[1]/self.ipred_ratio)
        idx_loc = 0
        for ipred in ipred_unique:
            # print(ipred+1)
            posx = int(img_final.size[0]*4/10) + 150 * numpy.floor(idx_loc/self.ipred_ratio)
            posy = img_final.size[1] + ipred_dist * (idx_loc % self.ipred_ratio) + ipred_dist/2
            # print(off_text,ipred_dist)
            # print(idx_loc%ipred_ratio)
            # print(numpy.floor(idx_loc/ipred_ratio))
            # if posy >= img_final.size[1]*2:
            #     posx = int(img_final.size[0]*4/10) + 100 * numpy.floor(off_text/ipred_dist)
            #     posy = img_final.size[1]+(off_text%ipred_dist)
            img_edit.text((posx, posy), self.names[ipred+1], text_color, font=self.fnt, anchor="ls")
            img_edit.rectangle((posx-30, posy-20, posx-10, posy), fill=(self.colors[ipred][0], self.colors[ipred][1], self.colors[ipred][2]), outline=(255, 255, 255))
            idx_loc += 1

        ipred_unique = numpy.unique(pred_sphe[:, :], return_counts=True)[0]
        idx_loc = 0
        for ipred in ipred_unique:
            # print(ipred+1)
            posx = int(img_final.size[0]*9/10) + 150 * numpy.floor(idx_loc/self.ipred_ratio)
            posy = img_final.size[1] + ipred_dist * (idx_loc % self.ipred_ratio) + ipred_dist/2
            img_edit.text((posx, posy), self.names[ipred+1], text_color, font=self.fnt, anchor="ls")
            img_edit.rectangle((posx-30, posy-20, posx-10, posy), fill=(self.colors[ipred][0], self.colors[ipred][1], self.colors[ipred][2]), outline=(255, 255, 255))
            idx_loc += 1

        save_dir = os.path.join(self.savedir, model_version)
        os.makedirs(save_dir, exist_ok=True)
        new_im.save(os.path.join(save_dir,  name_img + '_pred.png'))

    def save_all_2_nogt(self, img_orig, pred_persp, pred_sphe, model_version):
        name_img = (img_orig.split('/')[-1])[0:-8]

        # colorize prediction
        pred_persp_color = colorEncode(pred_persp, self.colors).astype(numpy.uint8)
        # pred_persp_color = (pred_persp).astype(numpy.uint8)
        pred_sphe_color = colorEncode(pred_sphe, self.colors).astype(numpy.uint8)

        # aggregate images and save
        im_vis = numpy.concatenate((pred_persp_color, pred_sphe_color), axis=1)
        img_final = PIL.Image.fromarray(im_vis)

        new_im = PIL.Image.new('RGB', (img_final.size[0], 2*img_final.size[1]))

        new_im.paste(PIL.Image.open(img_orig))
        # gt_image = os.path.join(self.datadir, name_img+'_seg.png')
        # new_im.paste(PIL.Image.open(gt_image), (int(img_final.size[0]/2), 0))
        new_im.paste(img_final, (0, img_final.size[1]))

        img_edit = ImageDraw.Draw(new_im)
        text_color = (255, 255, 255)

        ipred_unique = numpy.unique(pred_persp[:, :], return_counts=True)[0]
        ipred_ratio = 10
        ipred_dist = int(img_final.size[1]/ipred_ratio)
        idx_loc = 0
        for ipred in ipred_unique:
            # print(ipred+1)
            posx = int(img_final.size[0]*4/10) + 150 * numpy.floor(idx_loc/ipred_ratio)
            posy = img_final.size[1] + ipred_dist * (idx_loc % ipred_ratio) + ipred_dist/2
            # print(off_text,ipred_dist)
            # print(idx_loc%ipred_ratio)
            # print(numpy.floor(idx_loc/ipred_ratio))
            # if posy >= img_final.size[1]*2:
            #     posx = int(img_final.size[0]*4/10) + 100 * numpy.floor(off_text/ipred_dist)
            #     posy = img_final.size[1]+(off_text%ipred_dist)
            img_edit.text((posx, posy), self.names[ipred+1], text_color, font=self.fnt, anchor="ls")
            img_edit.rectangle((posx-30, posy-20, posx-10, posy), fill=(self.colors[ipred][0], self.colors[ipred][1], self.colors[ipred][2]), outline=(255, 255, 255))
            idx_loc += 1

        ipred_unique = numpy.unique(pred_sphe[:, :], return_counts=True)[0]
        idx_loc = 0
        for ipred in ipred_unique:
            # print(ipred+1)
            posx = int(img_final.size[0]*9/10) + 150 * numpy.floor(idx_loc/ipred_ratio)
            posy = img_final.size[1] + ipred_dist * (idx_loc % ipred_ratio) + ipred_dist/2
            img_edit.text((posx, posy), self.names[ipred+1], text_color, font=self.fnt, anchor="ls")
            img_edit.rectangle((posx-30, posy-20, posx-10, posy), fill=(self.colors[ipred][0], self.colors[ipred][1], self.colors[ipred][2]), outline=(255, 255, 255))
            idx_loc += 1

        save_dir = os.path.join(self.savedir, model_version)
        os.makedirs(save_dir, exist_ok=True)
        new_im.save(os.path.join(save_dir,  name_img + '_pred.png'))

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


class semseg_metric():
    def __init__(self):
        self.acc_meter = AverageMeter()
        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.time_meter = AverageMeter()

    def update_metrics(self, pred_seg_tmp, semseg_gt_id_tmp, timetic):
        self.time_meter.update(timetic)
        acc, pix = accuracy(pred_seg_tmp, semseg_gt_id_tmp)
        intersection, union = intersectionAndUnion(pred_seg_tmp, semseg_gt_id_tmp, 150)
        self.acc_meter.update(acc, pix)
        self.intersection_meter.update(intersection)
        self.union_meter.update(union)

    def get_iou(self):
        iou = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
        return iou

    def get_miou(self):
        iou = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
        return iou[np.nonzero(iou)].mean()

    def show_metrics(self):
        miou = self.get_miou()
        # for i, _iou in enumerate(iou):
        #     print('class [{}], IoU: {:.8f}'.format(i, _iou))

        print('[Eval Summary]:')
        print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
              .format(miou, self.acc_meter.average()*100, self.time_meter.average()))
        print()


def iou_mean(pred, target, n_classes=1):
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
    for iclass in range(1, n_classes+1):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == iclass
        target_inds = target == iclass
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

    print("Nombre images: ", len(OSS.list_img))

    if IMODE == "test":
        for elt in OSS.list_img:
            torch.cuda.synchronize()
            tic = time.perf_counter()

            pred_sphe, pred_persp = OSS.semseg_pred(elt)

            time_end = time.perf_counter() - tic
            # if VERBOSE:
            print("Done for ", str(elt), "in ", time_end)
            # OSS.save_simple(elt, pred_persp, pred_sphe)
            # OSS.save_all(elt, pred_persp, pred_sphe)
            OSS.save_all_2_nogt(elt, pred_persp, pred_sphe, model_version)

    elif IMODE == "infer":

        for elt in OSS.list_img:
            torch.cuda.synchronize()

            print("Doing for ", str(elt))
            pred_seg = OSS.semseg_single_pred(elt)
            OSS.save_single_nogt(elt, pred_seg, model_version)
            # print(numpy.unique(pred_seg, return_counts=True))

    elif IMODE == "eval":

        nb_classes_in_dataset = 150

        iou_glob_list = np.empty(5)
        iou_loc_list = np.empty(5)
        list_elt = np.array([])

        semseg_metric_glob = semseg_metric()

        for elt in OSS.list_img:
            torch.cuda.synchronize()
            tic = time.perf_counter()

            semseg_metric_loc = semseg_metric()

            semseg_gt_file = elt.replace("_rgb.png", "_seg.png")
            semseg_gt = as_numpy(PIL.Image.open(semseg_gt_file).convert('RGB'))

            semseg_gt_id = numpy.zeros((semseg_gt.shape[0], semseg_gt.shape[1]))
            # semseg_gt_id = numpy.zeros((semseg_gt.shape[0],semseg_gt.shape[1])) - 1
            for idx in range(semseg_gt.shape[0]):
                for idy in range(semseg_gt.shape[1]):
                    for idc, col in enumerate(OSS.colors):
                        if not((semseg_gt[idx, idy] - col).all()):
                            semseg_gt_id[idx, idy] = idc
                            break

            print("Doing for ", str(elt))
            pred_seg = OSS.semseg_single_pred(elt)
            OSS.save_single_nogt(elt, pred_seg, model_version)

            # print(numpy.unique(semseg_gt_id, return_counts=True))
            # print(numpy.unique(pred_persp, return_counts=True))

            pred_seg_tmp = pred_seg
            semseg_gt_id_tmp = semseg_gt_id
            if TOP_HALF:
                print("CAREFUL TOP HALF REDUCTION IS ACTIV!!")
                half_height = int(pred_seg_tmp.shape[0] / 2)
                # print(half_height)
                pred_seg_tmp = pred_seg_tmp[:half_height, :]
                semseg_gt_id_tmp = semseg_gt_id_tmp[:half_height, :]

            acc = np.mean((pred_seg_tmp == semseg_gt_id_tmp))

            # pred_persp_color = colorEncode(pred_seg_tmp, OSS.colors).astype(numpy.uint8)
            semseg_metric_glob.update_metrics(pred_seg_tmp, semseg_gt_id_tmp, time.perf_counter() - tic)
            iou_glob = semseg_metric_glob.get_iou()
            tmp_iou_glob_vector = [semseg_metric_glob.get_miou(), iou_glob[id_trees], iou_glob[id_earth], iou_glob[id_sky], acc]
            iou_glob_list = np.vstack((iou_glob_list, tmp_iou_glob_vector))

            semseg_metric_loc.update_metrics(pred_seg_tmp, semseg_gt_id_tmp, time.perf_counter() - tic)
            iou_loc = semseg_metric_loc.get_iou()
            tmp_iou_loc_vector = [semseg_metric_loc.get_miou(), iou_loc[id_trees], iou_loc[id_earth], iou_loc[id_sky], acc]
            iou_loc_list = np.vstack((iou_loc_list, tmp_iou_loc_vector))
            list_elt = np.append(list_elt, str(elt))

        # semseg_metric_glob.show_metrics()
        iou_glob_list = iou_glob_list[1:, :]
        iou_loc_list = iou_loc_list[1:, :]
        print("FROM GLOB MIOU {} iou_trees {} iou_ground {} iou_sky {} Acc {}".format(
            iou_glob_list[-1, 0], iou_glob_list[-1, 1], iou_glob_list[-1, 2], iou_glob_list[-1, 3], np.mean(iou_glob_list[:, -1])))
        print("FROM LOC MIOU {} iou_trees {} iou_ground {} iou_sky {} Acc {}".format(np.mean(iou_loc_list[:, 0]), np.mean(
            iou_loc_list[:, 1]), np.mean(iou_loc_list[:, 2]), np.mean(iou_loc_list[:, 3]), np.mean(iou_loc_list[:, -1])))

        np.savetxt(os.path.join(OSS.savedir, model_version) + "/iou_glob_list.csv", iou_glob_list, delimiter=",")
        np.savetxt(os.path.join(OSS.savedir, model_version) + "/iou_loc_list.csv", iou_loc_list, delimiter=",")
        np.savetxt(os.path.join(OSS.savedir, model_version) + "/list_elt.csv", list_elt, delimiter=",", newline="\n", fmt="%s")
        # np.savetxt(os.path.join(OSS.savedir, model_version) + "/miou_list_v2.csv", miou_list_v2[1:, :], delimiter=",")
        # np.savetxt(os.path.join(OSS.savedir, model_version) + "/iou_list_v2.csv", iou_list_v2[1:, :], delimiter=",")

        # print("KERAS v1 GMIOU : {}, giou : {}, MIOU: {} Acc : {}".format(giou_list[-1, 0], giou_list[-1, 1:], np.mean(iou_list[1:, 0]), np.mean(acc_list)))
        # print("KERAS v2 GMIOU : {}, giou : {}, MIOU: {} Acc : {}".format(0, [np.mean(iou_list_v2[0, 1:]),
        # np.mean(iou_list_v2[1, 1:]), np.mean(iou_list_v2[2, 1:])], np.mean(miou_list_v2[:, -1]), np.mean(miou_list_v2[:, 0])))

    elif IMODE == "compare":

        # semseg_metric_persp = semseg_metric()
        # semseg_metric_sphe = semseg_metric()

        nb_classes_in_dataset = 150

        iou_vector, acc_list, giou_list, iou_list = init_metrics(nb_classes_in_dataset)
        iou_vector_sphe, acc_list_sphe, giou_list_sphe, iou_list_sphe = init_metrics(nb_classes_in_dataset)

        for elt in OSS.list_img:

            semseg_gt_file = elt.replace("_rgb.png", "_seg.png")
            semseg_gt = as_numpy(PIL.Image.open(semseg_gt_file).convert('RGB'))
            # print("Image seg GT")
            # # print(semseg_gt)
            # print(numpy.unique(semseg_gt[:,:,0], return_counts=True)) #red
            # print(numpy.unique(semseg_gt[:,:,1], return_counts=True)) #green
            # print(numpy.unique(semseg_gt[:,:,2], return_counts=True)) #blue

            semseg_gt_id = numpy.zeros((semseg_gt.shape[0], semseg_gt.shape[1]))
            # semseg_gt_id = numpy.zeros((semseg_gt.shape[0],semseg_gt.shape[1])) - 1
            for idx in range(semseg_gt.shape[0]):
                for idy in range(semseg_gt.shape[1]):
                    for idc, col in enumerate(OSS.colors):
                        if not((semseg_gt[idx, idy] - col).all()):
                            semseg_gt_id[idx, idy] = idc
                            break
            # print("Semseg Gt ID")
            # print(semseg_gt_id)

            torch.cuda.synchronize()
            tic = time.perf_counter()

            # if VERBOSE:
            print("Doing for ", str(elt))
            pred_sphe, pred_persp = OSS.semseg_pred(elt)
            # OSS.save_all(elt, pred_persp, pred_sphe)
            OSS.save_all_2(elt, pred_persp, pred_sphe, model_version)

            # pred_sphe_color = colorEncode(pred_sphe, OSS.colors).astype(numpy.uint8)
            # pred_persp_color = colorEncode(pred_persp, OSS.colors).astype(numpy.uint8)

            # semseg_metric_persp.update_metrics(pred_persp_color, semseg_gt, time.perf_counter() - tic)
            # semseg_metric_sphe.update_metrics(pred_sphe_color, semseg_gt, time.perf_counter() - tic)

            print(numpy.unique(semseg_gt_id, return_counts=True))
            print(numpy.unique(pred_persp, return_counts=True))

            iou_vector, acc_list, giou_list, iou_list = update_metrics(semseg_gt_id, pred_persp, iou_vector, acc_list, giou_list, iou_list)
            iou_vector_sphe, acc_list_sphe, giou_list_sphe, iou_list_sphe = update_metrics(semseg_gt_id, pred_sphe, iou_vector_sphe, acc_list_sphe, giou_list_sphe, iou_list_sphe)

            # print("MIOU KERAS v0 : ", iou_mean(pred_sphe, semseg_gt_id, 150))
            # print("iou KERAS v1 : ", [m00.result().numpy(), m10.result().numpy(), m20.result().numpy()])

        # semseg_metric_persp.show_metrics("PERSP")
        # semseg_metric_sphe.show_metrics("SPHE")

        # np.savetxt("UNREAL_ENGINE/MAP_LAYOUT/LIST_GOALS/test_ig_RDMAPL_d20.csv", list_init_goal, delimiter=",")

        # tmp_iou = [m0.result().numpy(), m1.result().numpy(), m2.result().numpy()]

        # iou_vector = [gmiou, giou_trees, giou_ground, giou_sky, miou, iou_trees, iou_ground, iou_sky]
        # iou_vector_numpy = [iou_vector.m]

        np.savetxt(os.path.join(OSS.savedir, model_version) + "/giou_list.csv", giou_list[1:, :], delimiter=",")
        np.savetxt(os.path.join(OSS.savedir, model_version) + "/iou_list.csv", iou_list[1:, :], delimiter=",")
        np.savetxt(os.path.join(OSS.savedir, model_version) + "/giou_list_sphe.csv", giou_list_sphe[1:, :], delimiter=",")
        np.savetxt(os.path.join(OSS.savedir, model_version) + "/iou_list_sphe.csv", iou_list_sphe[1:, :], delimiter=",")

        print("KERAS v1 GMIOU : {}, giou : {}, MIOU: {} Acc : {}".format(giou_list[-1, 0], giou_list[-1, 1:], np.mean(iou_list[:, 0]), np.mean(acc_list)))
        print("KERAS v1 SPHERICAL GMIOU : {}, giou : {}, MIOU: {} Acc : {}".format(giou_list_sphe[-1, 0], giou_list_sphe[-1, 1:], np.mean(iou_list_sphe[:, 0]), np.mean(acc_list_sphe)))

    print("DONE")


if __name__ == '__main__':
    args = PARSER.parse_args()
    DATADIR = args.datadir
    SAVEDIR = args.savedir
    IMODE = args.mode
    TOP_HALF = args.top_half
    VERBOSE = args.VERBOSE is not None
    main()
