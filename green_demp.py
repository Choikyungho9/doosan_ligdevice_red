import argparse
import time
from pathlib import Path
import math
import PIL
import cv2
import torch
import torch.backends.cudnn as cudnn
import os
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box, plot_one_box_center
from utils.torch_utils import select_device, load_classifier, time_synchronized
from pygame import Surface
import numpy as np
import pygame
from pygame.locals import QUIT,Rect
os.environ['SDL_VIDEO_WINDOW_POS']=str('0,-10')
pygame.init()
pygame.display.set_caption('Control panel of pallet hall recogntion system')
BLACK=(0,0,0)
RED=(255,0,0)
GREEN=(0,255,0)
BLUE=(0,0,255)
ORANGE=(255,165,0)
WHITE=(255,255,255)
ang=90

# def monitor_init():
#     global screen, sub1, sub2, sub3, sub4, sub5
#     screen = pygame.display.set_mode([800, 600])
#     # Image(Surface) which will be refrenced
#     monitor = pygame.Surface((800, 600))
#
#     # Monitor rectangles for sections of  the monitor
#     mon1 = pygame.Rect(0, 0, 400, 250)
#     mon2 = pygame.Rect(400, 0, 400, 250)
#     mon3 = pygame.Rect(0, 250, 400, 250)
#     mon4 = pygame.Rect(400, 250, 400, 250)
#     mon5 = pygame.Rect(0, 500, 800, 100)
#
#     # subsurfaces of monitor
#     # Note that subx needs refreshing when monitor changes.
#     # pygame.draw.rect(screen, RED, [0,0,400,300])
#     sub1 = monitor.subsurface(mon1)
#     sub2 = monitor.subsurface(mon2)
#     sub3 = monitor.subsurface(mon3)
#     sub4 = monitor.subsurface(mon4)
#     sub5 = monitor.subsurface(mon5)
#
#     # Drawing a line on each split "screen"
#     pygame.draw.rect(sub1, RED, [0, 0, 400, 250])
#     pygame.draw.rect(sub2, ORANGE, [0, 0, 400, 250])
#     pygame.draw.rect(sub3, GREEN, [0, 0, 400, 250])
#     pygame.draw.rect(sub4, BLUE, [0, 0, 400, 250])
#     pygame.draw.rect(sub5, BLACK, [0, 0, 800, 100])
#
#     # monitor1's view  to the top left corner
#     screen.blit(sub1, (0, 0))
#     info11 = pygame.font.SysFont(None, 70)
#     info12 = pygame.font.SysFont(None, 70)
#     char11 = str("dist = ")
#     char12 = str("angle = ")
#     disp11 = info11.render(char11, True, BLACK)
#     disp12 = info12.render(char12, True, BLACK)
#     screen.blit(disp11, (20, 55))
#     screen.blit(disp12, (20, 125))
#     pygame.display.update()
#     # monitor2's view is in the top right corner
#     screen.blit(sub2, (400, 0))
#     info2 = pygame.font.SysFont(None, 80)
#     char = str("center : ")
#     disp2 = info2.render(char, True, BLACK)
#     screen.blit(disp2, (450, 75))
#     pygame.display.update()
#     # monitor3's view is in the bottom left corner
#     screen.blit(sub3, (0, 250))
#     info3 = pygame.font.SysFont(None, 65)
#     char1 = str("center x= ")
#     char2 = str('center y= ')
#
#     disp31 = info3.render(char1, True, BLACK)
#     disp32 = info3.render(char2, True, BLACK)
#     screen.blit(disp31, (20, 300))
#     screen.blit(disp32, (20, 380))
#     pygame.display.update()
#     # monitor4's view is in the bottom right corner
#     screen.blit(sub4, (400, 250))
#     info4 = pygame.font.SysFont(None, 80)
#     char = str("arrival : ")
#     disp4 = info4.render(char, True, BLACK)
#     screen.blit(disp4, (450, 350))
#     pygame.display.update()
#     # monitor5's view is in the bottom
#     screen.blit(sub5, (0, 500))
#     info5 = pygame.font.SysFont(None, 70)
#     char = str("speed = ")
#     disp5 = info5.render(char, True, BLACK)
#     screen.blit(disp5, (200, 520))
#     pygame.display.update()
#     # Update the screen
#     # pygame.display.update()
#     return

# def display(dist_m, angle_m, center_flag, center_x, center_y, arrival_flag, speed):
#     screen.blit(sub1, (0, 0))
#     info11 = pygame.font.SysFont(None, 70)
#     info12 = pygame.font.SysFont(None, 70)
#     dist = str("dist = ") + str(dist_m) + str(" mm")
#     angl = str("angle = ") + str(angle_m) + str(" deg")
#     disp11 = info11.render(dist, True, BLACK)
#     disp12 = info11.render(angl, True, BLACK)
#     screen.blit(disp11, (20, 55))
#     screen.blit(disp12, (20, 125))
#     pygame.display.update()
#
#     screen.blit(sub2, (400, 0))
#     info12 = pygame.font.SysFont(None, 80)
#     if center_flag == False:
#         dist21 = str("center : ")
#         disp2 = info12.render(dist21, True, BLACK)
#         screen.blit(disp2, (450, 75))
#     else:
#         dist22 = str("confirmed")
#         disp2 = info12.render(dist22, True, BLACK)
#         screen.blit(disp2, (450, 75))
#     pygame.display.update()
#
#     screen.blit(sub3, (0, 250))
#     info3 = pygame.font.SysFont(None, 65)
#     dist31 = str("center x= ") + str(center_x) + str(" mm")
#     dist32 = str('center y= ') + str(center_y) + str(" mm")
#     disp31 = info3.render(dist31, True, BLACK)
#     disp32 = info3.render(dist32, True, BLACK)
#     screen.blit(disp31, (20, 300))
#     screen.blit(disp32, (20, 380))
#     pygame.display.update()
#
#     screen.blit(sub4, (400, 250))
#     if arrival_flag == False:
#         info41 = pygame.font.SysFont(None, 80)
#         dist41 = str("arrival : ")
#         disp4 = info41.render(dist41, True, BLACK)
#         screen.blit(disp4, (450, 350))
#     else:
#         info42 = pygame.font.SysFont(None, 80)
#         dist42 = str("confirmed")
#         disp4 = info42.render(dist42, True, BLACK)
#         screen.blit(disp4, (450, 350))
#     pygame.display.update()
#
#     screen.blit(sub5, (0, 500))
#     info51 = pygame.font.SysFont(None, 70)
#     dist51 = str("speed = ") + str(speed) + str(" mm/s")
#     disp5 = info51.render(dist51, True, WHITE)
#     screen.blit(disp5, (200, 520))
#     pygame.display.update()
#
#     return


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def distance(x1, y1, x2, y2):
    """
    Calculate distance between two points
    """
    dist = math.sqrt(math.fabs(x2 - x1) ** 2 + math.fabs(y2 - y1) ** 2)
    return dist


@torch.no_grad()
def detect(weights='ligdevice.pt',  # model.pt path(s)
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           imgsz=640,  # inference size (pixels)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1000,  # maximum detections per image
           device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=False,  # show results
           save_txt=False,  # save results to *.txt
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           update=False,  # update all models
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           ):
    global pallet_centerx
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    count_img = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    x_top = float(xyxy[0])
                    y_top = float(xyxy[1])
                    width = float(xyxy[2] - xyxy[0])
                    height = float(xyxy[3] - xyxy[1])
                    pallet_x = float(xyxy[2])
                    pallet_y = float(xyxy[3])
                    # print(x_top," ",y_top, " ", width, " ", height)

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        (h, w) = im0.shape[:2]
                        cv2.circle(im0, (w // 2, h - 5), 7, (0, 255, 0), -3)
                        ref_w = w // 2
                        ref_h = h
                        #n_dist=0

                        if c == 1:  # pallet
                            distancei = (2 * 31.4 * 180) * 5 / (width + height)
                            #dist=dist+[int(distancei*10)/10]
                            #sum=0
                            #if n_dist > 20:
                            #    avg=dist[n_dist-20:n_dist]
                            #    avg1=[float(di) for di in avg]
                            #    for i in range(10):
                            #        sum=sum+avg1[i]
                            #    avg2=int(sum/20*10)/10
                            #    print(avg2)
                            plot_one_box(xyxy, im0, label=f'{distancei:.1f}cm', color=colors(c, True), line_thickness=2)
                            # x,y,w,h = cv2.boundingRect(xyxy)
                            # Draw circle in the center of the bounding box
                            pallet_x00 = int(x_top)  # + int(width / 2)
                            pallet_y00 = int(y_top)  # + int(height/2)
                            pallet_x01 = int(x_top + width)
                            pallet_y10 = int(y_top + height)
                            # x01_h = int(x_top + width * 0.295) # pallet hole1
                            # y10_h = int(y_top + height * 0.53) # pallet hole2
                            # x01_h2 = int(x_top + width * 0.705)
                            pallet_centerx = int((pallet_x01 + pallet_x00) / 2)
                            pallet_centery = int((pallet_y00 + pallet_y10) / 2)
                            #holex_c = int((x00 + x01))
                            #holey_c = int((y00 + y10))
                            cv2.circle(im0, (pallet_x00, pallet_y00), 4, (0, 255, 0), 0)  # 1-point
                            cv2.circle(im0, (pallet_x01, pallet_y00), 4, (0, 255, 0), 0)  # 2-point
                            cv2.circle(im0, (pallet_x00, pallet_y10), 4, (0, 255, 0), 0)  # 3-point
                            cv2.circle(im0, (pallet_x01, pallet_y10), 4, (0, 255, 0), 0)  # 4-point
                            # cv2.circle(im0, (x_center, y_center), 4, (0, 255, 0), -1) #center-point
                            # cv2.circle(im0, (x01_h, y10_h), 4, (0, 255, 0), -1)  # hole-center-point1
                            # cv2.circle(im0, (x01_h2, y10_h), 4, (0, 255, 0), -1)  # hole-center-point2
                            # A = (ref_w,ref_h)
                            # B = (x_center, y_center)
                            # -------------------------------------------------------------------------------------------------
                            # trig stuff to get the line
                            hypotenuse = distance(ref_w, ref_h, pallet_centerx, pallet_centery)
                            horizontal = distance(ref_w, ref_h, pallet_centerx, ref_h)
                            vertical = distance(pallet_centerx, pallet_centery, pallet_centerx, ref_h)
                            angle = np.arcsin(vertical / hypotenuse) * 180.0 / math.pi

                            # draw all 3 lines
                            cv2.line(im0, (ref_w, ref_h), (pallet_centerx, pallet_centery), (0, 0, 255), 2)
                            cv2.line(im0, (ref_w, ref_h), (pallet_centerx, ref_h), (0, 0, 255), 2)
                            cv2.line(im0, (pallet_centerx, pallet_centery), (pallet_centerx, ref_h), (0, 0, 255), 2)

                            # put angle text (allow for calculations upto 180 degrees)
                            angle_text = "degree"
                            if pallet_centery < ref_h and pallet_centerx > ref_w:
                                angle_text = str(int(angle))
                            elif pallet_centery < ref_h and pallet_centerx < ref_w:
                                angle_text = str(~int(angle))
                            elif pallet_centery > ref_h and pallet_centerx < ref_w:
                                angle_text = str(int(180 + angle))
                            elif pallet_centery > ref_h and pallet_centerx > ref_w:
                                angle_text = str(int(360 - angle))

                            # CHANGE FONT HERE
                            cv2.putText(im0, angle_text, (ref_w - 30, ref_h), cv2.FONT_HERSHEY_COMPLEX, 3,
                                        (0, 128, 229), 2)
                            # ------------------------------------------------------------------------------------------------
                        if c == 2:  # pallet_hole
                            # x,y,w,h = cv2.boundingRect(xyxy)
                            # Draw circle in the center of the bounding box
                            hole_x00 = int(x_top)  # + int(width / 2)
                            hole_y00 = int(y_top)  # + int(height/2)
                            hole_x01 = int(x_top + width)
                            hole_y10 = int(y_top + height)
                            # x01_h = int(x_top + width * 0.295)
                            # y10_h = int(y_top + height * 0.53)
                            # x01_h2 = int(x_top + width * 0.705)
                            hole_centerx = int((hole_x01 + hole_x00) / 2)
                            hole_centery = int((hole_y00 + hole_y10) / 2)
                            if hole_centerx < pallet_centerx :
                                ### pallet value
                                known_palletx_value = 235
                                known_pallety_value = 60
                                plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=2)
                                cv2.circle(im0, (hole_x00, hole_y00), 1, (0, 255, 0), 0)  # 1-point
                                cv2.circle(im0, (hole_x00, hole_y10), 1, (0, 255, 0), 0)  # 2-point
                                cv2.circle(im0, (hole_x01, hole_y00), 1, (0, 255, 0), 0)  # 3-point
                                cv2.circle(im0, (hole_x01, hole_y10), 1, (0, 255, 0), 0)  # 4-point
                                cv2.circle(im0, (hole_centerx, hole_centery), 5, (0, 255, 0), -1)  # center-point
                                # cv2.circle(im0, (x01_h, y10_h), 4, (0, 255, 0), -1)  # hole-center-point
                                # cv2.circle(im0, (x01_h2, y10_h), 4, (0, 255, 0), -1)  # hole-center-point
                                #print(hole_centerx, hole_centery, '1')
                                #print(pallet_centerx, pallet_centery, '2')
                                #print(pallet_y10)
                                dist_x = str(pallet_centerx - hole_centerx)
                                dist_y = str(pallet_y10 - hole_centery)
                                cv2.line(im0, (hole_centerx, hole_centery), (pallet_centerx, pallet_centery), (30, 255, 120), 2)
                                cv2.line(im0, (hole_centerx, hole_centery), (hole_centerx, pallet_y10), (30, 255, 120), 2)
                                #print('distance=',dist_x,dist_y)
                                x_dist = known_palletx_value
                                y_dist = known_pallety_value
                                estimated_xrate = float((x_dist / float(dist_x)))
                                estimated_yrate = float(y_dist / float(dist_y))
                                ####--- estimate distance x and y and multiply average value
                                # ***Correction of numerical values is required. ***
                                estimated_xdistance = float(float(estimated_xrate) * 170.2150961)
                                estimated_ydistance = float(float(estimated_yrate) * 92.18022163)
                                ####--- x and y rate => x/y value
                                # ***Correction of numerical values is required. ***
                                xyrate = estimated_xdistance / estimated_ydistance
                                estimated_real_xdistance = '%0.3f' % float(xyrate * 64.13288298)
                                estimated_real_ydistance = '%0.3f' % float(xyrate * 14.52065275)
                                #print(estimated_real_xdistance , estimated_real_ydistance)
                                cv2.putText(im0, f'{estimated_real_xdistance},{estimated_real_ydistance}', (ref_w - 550, ref_h - 550),
                                            cv2.FONT_HERSHEY_COMPLEX, 1,
                                            (224, 128, 229), 3)
#                                display(0, 0, False, estimated_real_xdistance,
 #                                       estimated_real_ydistance, True, 0)
                                img22 = PIL.Image.fromarray(im0, "RGB")
                                fn = "pic/file" + str(count_img) + ".jpg"
                                img22.save(fn)
                                count_img += 1
                                print(estimated_real_xdistance, estimated_real_ydistance,count_img)
                        else:
                            pass

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                #count += 1
            #  Printtime (inference + NMS)
            #print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    #print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='ligdevice.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=300, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    #print(opt)
    check_requirements(exclude=('tensorboard', 'thop'))

    detect(**vars(opt))
