import detectron2.data.datasets  # noqa # add pre-defined metadata
from lmpm.data.datasets.mevis import load_mevis_json
import json
import os
import pycocotools.mask as mask_util
import cv2
import numpy as np

image_root = "./datasets/mevis/valid_u/JPEGImages"
json_file = "./datasets/mevis/valid_u/meta_expressions.json"
anno_json = json.load(open(json_file, "r"))
dicts = load_mevis_json(image_root, json_file)  
mask_json = json.load(open("./datasets/mevis/valid_u/mask_dict.json", "r"))
base_dir = os.path.join('/workspace/datasets', 'mevis_vis')
os.makedirs(base_dir, exist_ok=True)

all_video_ids = list(anno_json['videos'].keys())

for video_id in all_video_ids:

    print("***********************")
    print(video_id)
    ann = anno_json['videos'][video_id]
    frame_paths = [os.path.join(image_root, video_id, frame_name+'.jpg') for frame_name in ann['frames']]
    H, W = cv2.imread(frame_paths[0]).shape[:2]
    exp_ann = ann['expressions']

    for i in list(exp_ann.keys()):
        exp = exp_ann[i]['exp']
        print(exp)
        exp_sentence = "_".join(exp.strip('., ').split())
        obj_id = exp_ann[i]['obj_id']
        anno_id = exp_ann[i]['anno_id']
        
        combined_mask = None
        for k, id in enumerate(anno_id):
            all_frame_mask = mask_json[str(id)]

            # all_frame_mask = [f if f is not None else {"size": [H,W], "counts": ""} for f in all_frame_mask]
            # print(all_frame_mask[0])
            all_frame_mask = [mask_util.decode(f) if f is not None else np.zeros((H, W), dtype=np.uint8) for f in all_frame_mask]

            all_frame_mask = np.stack(all_frame_mask, axis=2)
            if combined_mask is None:
                combined_mask = all_frame_mask
            else:
                combined_mask = np.logical_or(combined_mask, all_frame_mask)
        assert (combined_mask.shape[2] == len(frame_paths)), "mask shape not equal to frame number"
        for j, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            mask = combined_mask[:, :, j]
            alpha = 0.5
            
            color = [100, 0, 200] 
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0.5] = color

            # expanded_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            blended_image = alpha * frame + (1 - alpha) * colored_mask
            # resize to smaller size 1/4
            blended_image = cv2.resize(blended_image, (frame.shape[1]//4, frame.shape[0]//4))
            os.makedirs(os.path.join("/workspace/datasets", "mevis_vis", video_id, exp_sentence), exist_ok=True)
            path = os.path.join("/workspace/datasets", "mevis_vis", video_id, exp_sentence, str(j).zfill(4)+".jpg")
            cv2.imwrite(path, blended_image)
    