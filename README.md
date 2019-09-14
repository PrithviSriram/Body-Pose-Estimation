## Results

<p align="left">
<img src="https://github.com/PrithviSriram/Body-Pose-Estimation/blob/master/squat_multi.gif", width="720">
</p>

## Live Screen Results

<p align="left">
<img src="https://github.com/PrithviSriram/Body-Pose-Estimation/blob/master/test_1.gif", width="720">
</p>

## Instructions to run the code
1. Follow the steps in both the folders README to get the respective libraries codes running.
2. First run `python3 web_demo.py` in the `multi-body-pose-estimation` folder to get skeleton. 
3. Copy the output file to the `detecton` folder.
4. Run the following to get ghost filter. Once again, edit the file to set input name accordingly.

```
python tools/infer_simple.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    --output-dir /tmp/detectron-visualizations \
    --image-ext jpg \
    --wts https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
    demo
```
## Live Screen
1. Run `python3 live_screen.py` to get the object on live screen. Press `q` to quit once running.

## Remarks
1. Before executing any program, open it and change the input and output video path/name accordingly.
2. Detectron support is for python 2, beware of that.
3. The main alpha blending code is under the function vis_mask in the file `detectron/detectron/utils/vis.py`. 
