import cv2
import numpy as np
import os
import argparse
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]


def main():

    parser = argparse.ArgumentParser(description='Convert WeBots simulation snippets to video')
    parser.add_argument('--user', nargs='*', type=str, help='User for video perspective')
    parser.add_argument('--movement', nargs='*', type=str, help='User optimized')
    parser.add_argument('--target', nargs='*', type=str, help='Movement target object')
    parser.add_argument('--configuration', nargs='*', type=str, help='Object Configuration')

    args = parser.parse_args()

    print('[IMAGE TO VIDEO] Loading photo stills of the movement')
    img_array = []
    img_files = os.listdir('C:/Users/Miguel/Documents/WeBots/multi_user_legibility/controllers/camera/' +
                           args.user[0] + '/')
    img_files.sort(key=natural_keys)

    size = ()
    print('[IMAGE TO VIDEO] Create list sequence from the images')
    for filename in img_files:
        img = cv2.imread('C:/Users/Miguel/Documents/WeBots/multi_user_legibility/controllers/camera/' +
                         args.user[0] + '/' + filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
          
    print('[IMAGE TO VIDEO] Creating video from the images')
    if len(size) > 0:
        out = cv2.VideoWriter('../data/videos/configuration_' + args.configuration[0] + '/' + args.user[0] + '/' +
                              args.user[0] + '_movement_' + args.target[0] + '_' + args.movement[0] + '.mp4',
                              cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    else:
        print('[IMAGE TO VIDEO] Error creating video, no frame size defined!!')


if __name__ == '__main__':
    main()