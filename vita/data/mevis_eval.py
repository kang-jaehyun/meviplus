import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from .datasets.ytvis_api.ytvos import YTVOS

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager

from torch.nn import functional as F
from detectron2.utils.memory import retry_if_cuda_oom
import math
from skimage.morphology import disk
import cv2

class MeViSEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warning(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)

        # json_file = PathManager.get_local_path(self._metadata.json_file)
        # with contextlib.redirect_stdout(io.StringIO()):
        #     self._ytvis_api = YTVOS(json_file)

    def reset(self):
        self._predictions = []
        self._Js = []
        self._Fs = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        # prediction = instances_to_coco_json_video(inputs, outputs)
        # self._predictions.extend(prediction)
        pred_masks = outputs["pred_masks"] # 1, T, H, W
        _, T, H, W = pred_masks.shape
        pred_masks = pred_masks.squeeze(0)
        
        if inputs[0]['split'] == 'test':
            video_id = inputs[0]['video_name']
            exp_id = inputs[0]['exp_id']
            video_path = os.path.join(self._output_dir, video_id, exp_id)
            os.makedirs(video_path, exist_ok=True)
            for i, m in enumerate(pred_masks.cpu().numpy().astype(np.float32)):
                cv2.imwrite(os.path.join(video_path, f'{str(i).zfill(5)}.png'), m * 255)
            
        else:
            target_masks = torch.cat(inputs[0]['gt_masks_merge'])[None].to(pred_masks) # 1, T, H, W
            target_masks = retry_if_cuda_oom(F.interpolate)(
                target_masks.to(dtype=torch.float16),
                size=(H,W),
                mode="nearest",
            ) # 1, T, H, W
            
            # reduce batch
            target_masks = target_masks.squeeze(0) 
            
            j = self.db_eval_iou(target_masks, pred_masks)
            f = self.db_eval_boundary(target_masks, pred_masks)
            
            self._Js.append(j.mean().cpu())
            self._Fs.append(f.mean())
        
        

    
    def db_eval_iou(self, annotation, segmentation, void_pixels=None):
        """ Compute region similarity as the Jaccard Index.
        Arguments:
            annotation   (torch.tensor): binary annotation   map.
            segmentation (torch.tensor): binary segmentation map.
            void_pixels  (torch.tensor): optional mask with void pixels

        Return:
            jaccard (float): region similarity
        """
        assert annotation.shape == segmentation.shape, \
            f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
            

        annotation = annotation.to(dtype=torch.bool)
        segmentation = segmentation.to(dtype=torch.bool)

        if void_pixels is not None:
            assert annotation.shape == void_pixels.shape, \
                f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
            void_pixels = void_pixels.to(dtype=torch.bool)
        else:
            void_pixels = torch.zeros_like(segmentation)

        # Intersection between all sets
        inters = torch.sum((segmentation & annotation) & torch.logical_not(void_pixels), axis=(-2, -1)) # B(1), T
        union = torch.sum((segmentation | annotation) & torch.logical_not(void_pixels), axis=(-2, -1)) # B(1), T

        j = inters / union
        j[union == 0] = 1
        
        # if j.ndim == 0:
        #     j = 1 if np.isclose(union, 0) else j
        # else:
        #     j[np.isclose(union, 0)] = 1
        
        return j


    def db_eval_boundary(self, annotation, segmentation, void_pixels=None, bound_th=0.008):
        assert annotation.shape == segmentation.shape
        
        if void_pixels is not None:
            assert annotation.shape == void_pixels.shape
        if annotation.ndim == 3:
            n_frames = annotation.shape[0]
            f_res = torch.zeros(n_frames)
            for frame_id in range(n_frames):
                void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
                f_res[frame_id] = self.f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
        elif annotation.ndim == 2:
            f_res = self.f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
        else:
            raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
        return f_res


    def f_measure(self, foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
        """
        Compute mean,recall and decay from per-frame evaluation.
        Calculates precision/recall for boundaries between foreground_mask and
        gt_mask using morphological operators to speed it up.

        Arguments:
            foreground_mask (ndarray): binary segmentation image.
            gt_mask         (ndarray): binary annotated image.
            void_pixels     (ndarray): optional mask with void pixels

        Returns:
            F_measure (float): boundaries F-measure
        """
        # assert np.atleast_3d(foreground_mask).shape[2] == 1
        if void_pixels is not None:
            void_pixels = void_pixels.to(dtype=torch.bool)
        else:
            void_pixels = torch.zeros_like(foreground_mask).to(dtype=torch.bool)

        H, W = foreground_mask.shape
        bound_pix = bound_th if bound_th >= 1 else \
            torch.ceil(bound_th * torch.linalg.norm(torch.tensor((H,W), dtype=torch.float), ord=2)).item()
        

        # Get the pixel boundaries of both masks
        fg_boundary = self._seg2bmap(foreground_mask * torch.logical_not(void_pixels))
        gt_boundary = self._seg2bmap(gt_mask * torch.logical_not(void_pixels))

        # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
        # fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
        with torch.no_grad():
            fg_dil = F.conv2d(
                        fg_boundary[None, None].to(dtype=torch.float), 
                        torch.tensor(disk(bound_pix), dtype=torch.float, device=fg_boundary.device)[None, None], 
                        padding=int(bound_pix)
                        ).squeeze().to(dtype=torch.bool)
            # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
            # gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
            gt_dil = F.conv2d(
                        gt_boundary[None, None].to(dtype=torch.float), 
                        torch.tensor(disk(bound_pix), dtype=torch.float, device=gt_boundary.device)[None, None], 
                        padding=int(bound_pix)
                        ).squeeze().to(dtype=torch.bool)


            #     # Get the intersection
        gt_match = gt_boundary * fg_dil
        fg_match = fg_boundary * gt_dil

        #     # Area of the intersection
        n_fg = torch.sum(fg_boundary)
        n_gt = torch.sum(gt_boundary)

        # % Compute precision and recall
        if n_fg == 0 and n_gt > 0:
            precision = 1
            recall = 0
        elif n_fg > 0 and n_gt == 0:
            precision = 0
            recall = 1
        elif n_fg == 0 and n_gt == 0:
            precision = 1
            recall = 1
        else:
            precision = torch.sum(fg_match) / float(n_fg)
            recall = torch.sum(gt_match) / float(n_gt)

        # Compute F measure
        if precision + recall == 0:
            F_measure = 0
        else:
            F_measure = 2 * precision * recall / (precision + recall)

        return F_measure


    def _seg2bmap(self, seg, width=None, height=None):
        """
        From a segmentation, compute a binary boundary map with 1 pixel wide
        boundaries.  The boundary pixels are offset by 1/2 pixel towards the
        origin from the actual segment boundary.
        Arguments:
            seg     : Segments labeled from 1..k.
            width	  :	Width of desired bmap  <= seg.shape[1]
            height  :	Height of desired bmap <= seg.shape[0]
        Returns:
            bmap (ndarray):	Binary boundary map.
        David Martin <dmartin@eecs.berkeley.edu>
        January 2003
        """

        seg = seg.to(dtype=torch.bool)
        seg[seg > 0] = 1

        # assert np.atleast_3d(seg).shape[2] == 1

        width = seg.shape[1] if width is None else width
        height = seg.shape[0] if height is None else height

        h, w = seg.shape[:2]

        ar1 = float(width) / float(height)
        ar2 = float(w) / float(h)

        assert not (
            width > w | height > h | abs(ar1 - ar2) > 0.01
        ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

        e = torch.zeros_like(seg)
        s = torch.zeros_like(seg)
        se = torch.zeros_like(seg)

        e[:, :-1] = seg[:, 1:]
        s[:-1, :] = seg[1:, :]
        se[:-1, :-1] = seg[1:, 1:]

        b = seg ^ e | seg ^ s | seg ^ se
        b[-1, :] = seg[-1, :] ^ e[-1, :]
        b[:, -1] = seg[:, -1] ^ s[:, -1]
        b[-1, -1] = 0

        if w == width and h == height:
            bmap = b
        else:
            bmap = torch.zeros((height, width))
            for x in range(w):
                for y in range(h):
                    if b[y, x]:
                        j = 1 + math.floor((y - 1) + height / h)
                        i = 1 + math.floor((x - 1) + width / h)
                        bmap[j, i] = 1

        return bmap


        
    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            Js = comm.gather(self._Js, dst=0)
            Fs = comm.gather(self._Fs, dst=0)
            
            Js = list(itertools.chain(*Js))
            Fs = list(itertools.chain(*Fs))
            
            # predictions = comm.gather(self._predictions, dst=0)
            # predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            Js = self._Js
            Fs = self._Fs
            # predictions = self._predictions

        if len(Js) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        # if self._output_dir:
        #     PathManager.mkdirs(self._output_dir)
        #     file_path = os.path.join(self._output_dir, "instances_predictions.pth")
        #     with PathManager.open(file_path, "wb") as f:
        #         torch.save(predictions, f)

        self._results = OrderedDict()
        
        results = {
            "J" : np.mean(Js),
            "F" : np.mean(Fs),
            "J&F" : (np.mean(Js) + np.mean(Fs))/2,
        }
        
        self._results.update(results)
        
        # self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for YTVIS format ...")

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in predictions:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(predictions))
                f.flush()

        self._logger.info("Annotations are not available for evaluation.")
        return


def instances_to_coco_json_video(inputs, outputs):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        video_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    assert len(inputs) == 1, "More than one inputs are loaded for inference!"

    video_id = inputs[0]["video_id"]

    scores = outputs["pred_scores"]
    labels = outputs["pred_labels"]
    masks = outputs["pred_masks"]

    ytvis_results = []
    for (s, l, m) in zip(scores, labels, masks):
        segms = [
            mask_util.encode(np.array(_mask[:, :, None], order="F", dtype="uint8"))[0]
            for _mask in m
        ]
        for rle in segms:
            rle["counts"] = rle["counts"].decode("utf-8")

        res = {
            "video_id": video_id,
            "score": s,
            "category_id": l,
            "segmentations": segms,
        }
        ytvis_results.append(res)

    return ytvis_results
