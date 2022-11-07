from ast import arg
import os
import sys
#from typing import final
import torch
import torch.nn as nn
# import tqdm
import cv2
import numpy as np
import ray
import copy
import skimage

sys.path.append('detectron2')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.structures import Instances
from detectron2.layers.nms import nms


from bottom_up_attention_pytorch.utils.utils import mkdir, save_features
from bottom_up_attention_pytorch.utils.extract_utils import get_image_blob, save_bbox, save_roi_features_by_bbox, save_roi_features
from bottom_up_attention_pytorch.utils.progress_bar import ProgressBar
from bottom_up_attention_pytorch.bua import add_config
from bottom_up_attention_pytorch.bua.caffe.modeling.box_regression import BUABoxes
from torch.nn import functional as F
from detectron2.modeling import postprocessing
from bottom_up_attention_pytorch.utils.extract_features_singlegpu import extract_feat_singlegpu_start
from bottom_up_attention_pytorch.utils.extract_features_multigpu import extract_feat_multigpu_start
from bottom_up_attention_pytorch.utils.extract_features_faster import extract_feat_faster_start
from bottom_up_attention_pytorch.bua.d2.modeling.roi_heads import AttributeROIHeads, AttributeRes5ROIHeads,register

from ray.actor import ActorHandle

class FeatureExtractor:
    def __init__(self): 
        # parser = argparse.ArgumentParser(description="PyTorch Object Detection2 Inference")
        # parser.add_argument(
        #     "--config-file",
        #     default="/home/harry/Desktop/Dat/drif-master/bottom_up_attention_pytorch/configs/d2/train-d2-r101.yaml",
        #     metavar="FILE",
        #     help="path to config file",
        # )

        # parser.add_argument('--num-cpus', default=1, type=int, 
        #                     help='number of cpus to use for ray, 0 means no limit')

        # parser.add_argument('--gpus', dest='gpu_id', help='GPU id(s) to use',
        #                     default='1', type=str)

        # parser.add_argument("--mode", default="d2", type=str, help="'caffe' and 'd2' indicates \
        #                     'use caffe model' and 'use detectron2 model'respectively")

        # parser.add_argument('--extract-mode', default='bbox_feats', type=str,
        #                   help="'roi_feats', 'bboxes' and 'bbox_feats' indicates \
        #                   'extract roi features directly', 'extract bboxes only' and \
        #                   'extract roi features with pre-computed bboxes' respectively")

        # parser.add_argument('--min-max-boxes', default='100', type=str, 
        #                     help='the number of min-max boxes of extractor')

        # # parser.add_argument('--out-dir', dest='output_dir',
        # #                     help='output directory for features',
        # #                     default="features")
        # parser.add_argument('--image-dir', dest='image_dir',
        #                     help='directory with images',
        #                     default="image")
        # parser.add_argument('--bbox-dir', dest='bbox_dir',
        #                     help='directory with bbox',
        #                     default="bbox")
        # parser.add_argument("--fastmode", action="store_true", help="whether to use multi cpus to extract faster.",)

        # parser.add_argument(
        #     "--resume",
        #     action="store_true",
        #     help="whether to attempt to resume from the checkpoint directory",
        # )
        # parser.add_argument(
        #     "opts",
        #     help="Modify config options using the command-line",
        #     default=None,
        #     nargs=argparse.REMAINDER,
        # )

        #self.args = parser.parse_args()
        self.config_file = '/home/harry/Desktop/Dat/drif-master/bottom_up_attention_pytorch/configs/d2/test-d2-r101.yaml'
        self.mode = 'd2'
        self.min_max_boxes = '100'
        ###TODO: Check the extract mode, maybe it's bbox_feats
        self.extract_mode = 'roi_feats'
        self.num_cpus = 1
        self.gpus = '1' 
        self.cfg = self.setup()
        self.resume = False
        self.model = None
        
    def model_inference(self, model, batched_inputs, extract_mode, image_h, image_w, attribute_on=False):#remove dump_folder
        # print(f"BATCHED_INPUTS: {batched_inputs}", len(batched_inputs))
        # batched_inputs_sample = batched_inputs[0]["image"] #torch.Size([128, 31, 1000])
        # print(f"BATCHED_INPUTS_SIZE: {batched_inputs_sample.size()}")
        images = model.preprocess_image(batched_inputs)
        features = model.backbone(images.tensor)
        # print(f"IMAGES:{images}\nFEATURES: {features}")
        if extract_mode != 3:
            proposals, _ = model.proposal_generator(images, features, None)
        else: # feats_by_box
            assert "proposals" in batched_inputs[0]
            print("proposals")
            proposals = [x["proposals"].to(model.device) for x in batched_inputs]
        _, pooled_features, _ = model.roi_heads.get_roi_features(features, proposals)  # fc7 feats
        predictions = model.roi_heads.box_predictor(pooled_features)
        cls_lables = torch.argmax(predictions[0], dim=1)
    
        cls_probs = F.softmax(predictions[0], dim=-1)
        cls_probs = cls_probs[:, :-1]  # background is last
        if extract_mode != 3:
            predictions, r_indices = model.roi_heads.box_predictor.inference(predictions, proposals)

            if attribute_on:
                attr_scores = model.roi_heads.forward_attribute_score(pooled_features, cls_lables)
                attr_probs = F.softmax(attr_scores, dim=-1)
                attr_probs = attr_probs[r_indices]

        # postprocess
            height = images[0].shape[1]
            width = images[0].shape[2]
            r = postprocessing.detector_postprocess(predictions[0], height, width) # image

            bboxes = r.get("pred_boxes").tensor  # box
            classes = r.get("pred_classes")  # classes
            cls_probs = cls_probs[r_indices]  # clsporbs

            pooled_features = pooled_features[r_indices]

            if extract_mode == 1: # roi_feats

                assert (
                    bboxes.size(0)
                    == classes.size(0)
                    == cls_probs.size(0)
                    == pooled_features.size(0)
                )
         
                oriboxes = bboxes / batched_inputs[0]['im_scale']
                if not attr_scores is None:
                    info = {
                        "objects": classes.cpu().numpy(),
                        "cls_prob": cls_probs.cpu().numpy(),
                        'attrs_id': attr_probs,
                        'attrs_scores': attr_scores,
                    }
                else:
                    # save info and features
                    info = {
                        "objects": classes.cpu().numpy(),
                        "cls_prob": cls_probs.cpu().numpy(),
                    }

                # np.savez_compressed(
                #     os.path.join(dump_folder), 
                #     x=pooled_features.cpu().numpy(), 
                #     bbox=oriboxes.cpu().numpy(), 
                #     num_bbox=oriboxes.size(0), 
                #     image_h=image_h,
                #     image_w=image_w,
                #     image_h_inner=r.image_size[0], 
                #     image_w_inner=r.image_size[1],
                #     info=info
                # )
            elif extract_mode == 2:  # bbox only
                oriboxes = bboxes / batched_inputs[0]['im_scale']
                # np.savez_compressed(
                #     os.path.join(dump_folder), 
                #     bbox=oriboxes.cpu().numpy(), 
                #     num_bbox=oriboxes.size(0)
                # )
            else:
                raise Exception("extract mode not supported:{}".format(extract_mode))
        
            if attribute_on:
                return [bboxes], [cls_probs], [pooled_features], [attr_probs]
            else:
                return [bboxes], [cls_probs], [pooled_features]
        else:  # extract mode == 3
            # postprocess
            height = images[0].shape[1]
            width = images[0].shape[2]

            if attribute_on:
                attr_scores = model.roi_heads.forward_attribute_score(pooled_features, cls_lables)
                attr_probs = F.softmax(attr_scores, dim=-1)

            bboxes = batched_inputs[0]['proposals'].proposal_boxes.tensor
            oriboxes = bboxes / batched_inputs[0]['im_scale']

        # 保存特征
            if not attr_scores is None:
                info = {
                "objects": cls_lables.cpu().numpy(),
                "cls_prob": cls_probs.cpu().numpy(),
                'attrs_id': attr_probs,
                'attrs_conf': attr_scores,
                }
            else:
                info = {
                "objects": cls_lables.cpu().numpy(),
                "cls_prob": cls_probs.cpu().numpy(),
                }
            # np.savez_compressed(
            #     os.path.join(dump_folder), 
            #     x=pooled_features.cpu().numpy(), 
            #     bbox=oriboxes.cpu().numpy(), 
            #     num_bbox=oriboxes.size(0), 
            #     image_h=image_h,
            #     image_w=image_w,
            #     image_h_inner=height,
            #     image_w_inner=width,
            #     info=info
            # )
        
            if attribute_on:
                return [bboxes], [cls_probs], [pooled_features], [attr_probs]
            else:
                return [bboxes], [cls_probs], [pooled_features]
        
    
    def extract_feat(self, im, cfg, actor: ActorHandle):
        # num_images = len(img_list)
        #print('Number of images on split{}: {}.'.format(split_idx, num_images))
        if self.model == None:
            self.model = DefaultTrainer.build_model(cfg)
            DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=False
            )
            self.model.eval()
        register()
        print(f" EXTRACTOR_MODE: {cfg.MODEL.BUA.EXTRACTOR.MODE}")
        #for im_file in (img_list):
            # if os.path.exists(os.path.join(special_output_dir, im_file.split('.')[0]+'.npz')):
            #     actor.update.remote(1)
            #     continue
            #image_id = im_file.split('.')[0] # xxx.jpg
            #dump_folder = os.path.join(args.output_dir, str(image_id) + ".npz")
            #if os.path.exists(os.path.join(args.output_dir, im_file.split('.')[0]+'.npz')):
            #    actor.update.remote(1)
            #    continue
        # else:
        #     start = True
        # if not start:
        #     actor.update.remote(1)
        #     continue
        #    im = cv2.imread(os.path.join(args.image_dir, im_file))
        illegal = False
        if im is None:
            illegal = True
        elif im.shape[-1] != 3:
            illegal = True
        elif max(im.shape[:2]) / min(im.shape[:2]) > 10 or max(im.shape[:2]) < 25:
            illegal = True
        if illegal:
            # print(os.path.join(args.image_dir, im_file), "is illegal!")
            actor.update.remote(1)
        # dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
        pixel_mean = cfg.MODEL.PIXEL_MEAN if self.mode == "caffe" else 0.0
        image_h = np.size(im, 0)
        image_w = np.size(im, 1)
        dataset_dict = get_image_blob(im, pixel_mean)
        # extract roi features
        if cfg.MODEL.BUA.EXTRACTOR.MODE == 1:
            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = self.model_inference(self.model,[dataset_dict], 1, image_h, image_w,True)
                else:
                    boxes, scores, features_pooled = self.model_inference(self.model, [dataset_dict], 1, image_h, image_w)
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 2:
            with torch.set_grad_enabled(False):
                boxes, scores, _ = self.model_inference(self.model,[dataset_dict], 2, image_h, image_w)
        elif cfg.MODEL.BUA.EXTRACTOR.MODE == 3: # extract roi features by bbox
            # npy = False
            # if os.path.exists(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npy')):
            #     npy = True
            # elif not os.path.exists(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npz')):
            #     actor.update.remote(1)
            # if npy:
            #     try:
            #         bbox = torch.from_numpy(np.load(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npy'), allow_pickle=True).tolist()['bbox']) * dataset_dict['im_scale']
            #     except Exception as e:
            #         print(e)
            #         actor.update.remote(1)
            # else:
            #     bbox = torch.from_numpy(np.load(os.path.join(args.bbox_dir, im_file.split('.')[0]+'.npz'))['bbox']) * dataset_dict['im_scale']
            bbox, box_score, _  = self.model_inference(self.model,[dataset_dict], 2, image_h, image_w) * dataset_dict['im_scale']
            proposals = Instances(dataset_dict['image'].shape[-2:])
            proposals = Instances(dataset_dict['image'].shape[-2:])
            proposals.proposal_boxes = BUABoxes(bbox)
            dataset_dict['proposals'] = proposals

            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = self.model_inference(self.model,[dataset_dict], 3, image_h, image_w, True)
                else:
                    boxes, scores, features_pooled = self.model_inference(self.model,[dataset_dict], 3, image_h, image_w)
        else:
            raise Exception("extract mode not supported.")
        actor.update.remote(1)
        output = {
          "features": features_pooled,
          "bbox": boxes,
          "image_width": image_w,
          "image_height":image_h,
          "scores": scores
        }
        return output
    
    
    def sorting_features(self, output):
        """
        Output: a dictionary contain all features, box, img info, scores (torch.Tensor)
        output = {
          "features": features_pooled,
          "bbox": boxes,
          "image_width": image_w,
          "image_height":image_h,
          "scores": scores
        }
        
        Return a dictionary that only choose 5 features with highest scores
        """
        features = output["features"]
        boxes = output["bbox"]
        scores = output["scores"]
        image_w = output["image_width"]
        image_h = output["image_height"]
        
        #Loop throught the list of scores to save the index in a dictionary
        scores_dict = {}
        for i in range(len(scores[0])):
            scores_dict[scores[0][i]] = i
        
        sorted_scores = sorted(scores[0])
        sorted_features = torch.zeros(size=(5,2048))
        sorted_boxes = torch.zeros(size=(5,4))
        
        for i in range(5):
            #get index
            index = scores_dict[sorted_scores[i]]
            sorted_features[i,:] = features[index]
            sorted_boxes[i,:] = boxes[index]
        
        sorted_output = {
            "features": sorted_features,
            "bbox" : sorted_boxes,
            "image_width": image_w,
            "image_height": image_h,
            "scores": sorted_scores
        }
        
        return sorted_output
        
        
    
    def switch_extract_mode(self,mode):
        if mode == 'roi_feats':
            switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 1]
        elif mode == 'bboxes':
            switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 2]
        elif mode == 'bbox_feats':
            switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 3, 'MODEL.PROPOSAL_GENERATOR.NAME', 'PrecomputedProposals']
        else:
            print('Wrong extract mode! ')
            exit()
        return switch_cmd
  
  
    def set_min_max_boxes(self, min_max_boxes, mode):
        if min_max_boxes == 'min_max_default':
            return []
        try:
            min_boxes = int(min_max_boxes.split(',')[0])
            max_boxes = int(min_max_boxes.split(',')[1])
            if mode == "caffe":
                pass
            elif mode == "d2":
                if min_boxes == 100 & max_boxes == 100:
                    cmd = ['MODEL.BUA.EXTRACTOR.MIN_BOXES', min_boxes, 
                            'MODEL.BUA.EXTRACTOR.MAX_BOXES', max_boxes,
                            'MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.0,
                            'MODEL.ROI_HEADS.NMS_THRESH_TEST', 0.3 ]
                    return cmd
            else:
                raise Exception("detection mode not supported: {}".format(mode))
        except:
            print('Illegal min-max boxes setting, using config default. ')
            return []
        cmd = ['MODEL.BUA.EXTRACTOR.MIN_BOXES', min_boxes, 
                'MODEL.BUA.EXTRACTOR.MAX_BOXES', max_boxes]
        return cmd
  
  
    def setup(self):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        add_config(self.mode, cfg)
        cfg.merge_from_file(self.config_file)
        # cfg.merge_from_list(self.opts)
        cfg.merge_from_list(['MODEL.BUA.EXTRACT_FEATS',True])
        cfg.merge_from_list(self.switch_extract_mode(self.extract_mode))
        cfg.merge_from_list(self.set_min_max_boxes(self.min_max_boxes, self.mode))
        cfg.freeze()
        default_setup(cfg, self.config_file)
        return cfg
    
    def _process_feature_extraction(self, cfg, img, infos):#step 3
        '''
        #predictor.model.roi_heads.box_predictor.test_topk_per_image = 1000
        #predictor.model.roi_heads.box_predictor.test_nms_thresh = 0.99
        #predictor.model.roi_heads.box_predictor.test_score_thresh = 0.0
        #pred_boxes = [x.pred_boxes for x in instances]#can use prediction boxes
        '''
        pb = ProgressBar(1)
        #print(f"PROCSS_FEATURE_EXTRACTION_IM: {img}")
        actor = pb.actor
        instances = self.extract_feat(img, cfg, actor)
        feats = instances['features']
        result = {
            'bbox': instances['bbox'][0].cpu().numpy(),
            #'num_boxes' : len(instances),#len(pred_instances[0]['instances'].pred_boxes[pred_inds].tensor.cpu().numpy()),
            'width' : instances['image_width'],
            'height' : instances['image_height'],
            'cls_prob': instances['scores'],#pred_instances[0]['instances'].scores.cpu().numpy(),
            }
        return feats, result

    def get_detectron2_features(self, cfg, img):#step 2
        #we have to PREPROCESS the tensor before partially executing it!
        #taken from https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/defaults.py
        #images = []
        image_info = []
        height, width = img.shape[:2]
        # img = predictor.aug.get_transform(img).apply_image(img)
        # img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        # images.append({"image": img, "height": height, "width": width})
        image_info.append({"height": height, "width": width})
        #returns features and infos
        return self._process_feature_extraction( cfg, img, image_info)
      
      
    def extract_features_and_spatial_location(self, cfg, image):
        features, infos = self.get_detectron2_features(cfg, image)
        image_h = infos["height"]
        image_w = infos["width"]
        #num_boxes = infos["num_boxes"]
        boxes = infos["bbox"].reshape(-1,4)
        features = features[0].to(device = 'cuda')
        # print(f"FEATURES_SIZE:{features.size()}")
        
        features = features.reshape(-1, 2048).cpu().detach().numpy()#2048
        num_boxes = features.shape[0]

        g_feat = np.sum(features, axis=0)/num_boxes
        num_boxes +=1
        features = np.concatenate(
                        [np.expand_dims(g_feat, axis=0), features], axis=0
                    )
        
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:, :4] = boxes
        image_location[:, 4] = (
            (image_location[:, 3] - image_location[:, 1])
            * (image_location[:, 2] - image_location[:, 0])
            / (float(image_w) * float(image_h))
        )

        image_location_ori = copy.deepcopy(image_location)

        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)

        g_location = np.array([0, 0, 1, 1, 1])
        image_location = np.concatenate(
        [np.expand_dims(g_location, axis=0), image_location], axis=0
        )
        self.boxes = image_location

        g_location_ori = np.array(
            [0, 0, image_w, image_h, image_w * image_h]
        )
        image_location_ori = np.concatenate(
            [np.expand_dims(g_location_ori, axis=0), image_location_ori],
            axis=0,
        )
        self.boxes_ori = image_location_ori
        self.num_boxes = num_boxes
        # print(f"NUM_BOXES:{self.num_boxes}\n IMAGE_LOCATION:{image_location.shape}")
        
        return features, num_boxes, image_location, image_location_ori

    def padding(self, features, padding_dim):
        store_features = []
        """
        features: List of numpy array with different size
        padding_dim: final dimension of array after padding
        """
        for feat in features:
            padding = np.zeros((padding_dim, feat.shape[1]), dtype=np.float32)
            # print(f"FEAT_SHAPE: {feat.shape}\nPADDING_DIM: {padding_dim}")
            padding[:feat.shape[0],:] = feat
            padding = torch.Tensor(padding).float()
            store_features.append(padding)
            final_features = torch.stack([store_features[i] for i in range(len(store_features))], dim =0)

        return final_features
    
    def __getitem__(self, image, batch_size):
        cfg = self.cfg
        all_features_shape, all_spatials_shape, image_masks_shape =[], [], []
        features, spatials, image_masks = [], [], []
        for i in range(batch_size):
            img = image[i].detach().cpu().numpy()
            img = np.swapaxes(img,0,2) #(128,96,3)
            img= np.swapaxes(img,0,1) #(96,128,3)
            # print(f"INPUT_IMAGE_SHAPE: {img.shape}") #[600,800,3]
            feature, num_boxes, boxes, _ = self.extract_features_and_spatial_location(cfg, img)
            # print(f"NUM_BOXES:{num_boxes}\n NUMBER_OF_BOXES: {boxes.size}")
            mix_num_boxes = min(int(num_boxes), 37)
            mix_boxes_pad = np.zeros((mix_num_boxes, 5))
            mix_features_pad = np.zeros((mix_num_boxes, 2048))

            mix_num_boxes = int((mix_num_boxes+1)/2)
            image_mask = [1] * (int(mix_num_boxes))
            while len(image_mask) < mix_num_boxes:
                image_mask.append(0)
        
            mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
            mix_features_pad[:mix_num_boxes] = feature[:mix_num_boxes]

            # features = torch.tensor(mix_features_pad).float()
            # image_mask = torch.tensor(image_mask).long()
            # spatials = torch.tensor(mix_boxes_pad).float()
            all_features_shape.append(mix_features_pad.shape[0])
            all_spatials_shape.append(mix_boxes_pad.shape[0])
            image_masks_shape.append(len(image_mask))
            features.append(mix_features_pad)
            spatials.append(mix_boxes_pad)
            image_masks.append(image_mask)
        
        padding_dim = 14#max(all_features_shape)+1
        spatials_padding_dim = 14#max(all_spatials_shape)+1
        masks_padding_dim = max(image_masks_shape) + 1
        ###TODO(OKE): FIX THIS  USING TORCH>STACK TO ADD BATCH
        final_features = self.padding(features, padding_dim)#torch.tensor(features).float()
        final_spatials = self.padding(spatials, spatials_padding_dim)#torch.tensor(spatials).float()
        i = 0
        for masking in image_masks:
            masking = np.array(masking)
            padding = np.zeros(masks_padding_dim-masking.size)
            masking = np.concatenate([masking, padding])
            image_masks[i] = masking
            i+=1
        final_image_masks = torch.tensor(image_masks)
        

        return final_features, final_spatials, final_image_masks