import cv2
import numpy as np
import pandas as pd
import csv
import time


def gen_rows(stream, max_length=None):
    rows = csv.reader(stream)
    if max_length is None:
        rows = list(rows)
        max_length = max(len(row) for row in rows)
    for row in rows:
        yield row + [float('nan')] * (max_length - len(row))


if __name__ == "__main__":
    
    video_capture = cv2.VideoCapture(0) #ENTER FILE NAME HERE
    video_loop = cv2.VideoCapture('squat_red_f.avi')
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    f_out = cv2.VideoWriter('output.avi', fourcc, 20, (frame_width, frame_height))
    #df = pd.read_csv('./detectron/mask.csv', header=None)
    with open('./detectron/idx_25.csv') as f:
        df = pd.DataFrame.from_records(list(gen_rows(f)))
    i = 0
    start_time = time.time()
    flag = 1
    start = 1

    while(True):
        # Capture frame-by-frame
        ret, oriImg = video_capture.read()
        ret2, loopf = video_loop.read()
        if(flag and loopf is None):
            start_time = time.time()
            #i = 0
            flag = 0
        #if oriImg is None:
        #    break

        if(flag == 0 and time.time() - start_time >= 1):
            video_loop = cv2.VideoCapture('squat_red_f.avi')
            ret2, loopf = video_loop.read()
            i = 0
            flag = 1

        if(flag): 
            mask = df[i:i+1].values
            #print(mask)
            mask = mask.astype(np.float32)
            mask = mask[np.logical_not(np.isnan(mask))]
            mask = np.reshape(mask, (2, -1))
            #print(mask)
            #mask = np.reshape(mask, (2, ))
            mask = mask.astype(np.int32)
            #print(mask[0])
            #print(mask[1])
            #idx = np.nonzero(mask)
            idx = (mask[0], mask[1])
            #print(np.shape(idx))
            #print(idx[0])
            #print(idx[1])
            
            #loopf = loopf[idx[0], idx[1], :]
            #print(loopf.shape)

            fw = int(loopf.shape[0] / 2.5)
            fh = int(loopf.shape[1] / 2.5)
            #print(loopf.shape)
            #print(str(fw) + " " + str(fh))
            loopf = cv2.resize(loopf, (fh, fw))
            #print(loopf.shape)
            shape_dst = np.min(oriImg.shape[0:2])
            alpha = 0.4

            oriImg[idx[0], idx[1]-180, :] = oriImg[idx[0], idx[1]-80, :] * (alpha) + loopf[idx[0], idx[1], :] * (1.0 - alpha)

            # Display the resulting frame
            #print("End")
            #f_out.write(oriImg)
            i = i + 1

        cv2.imshow('Video', oriImg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()