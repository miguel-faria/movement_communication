import numpy as np
import os
import argparse

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def split_video(video_path, video_dir, vid_name, time_intervals, part_extensions):

    if len(time_intervals) != len(part_extensions):
        print('[VIDEO TRIMMING] Mismatch beteween video intervals and number of videos to trim.')
        return

    else:
        print('[VIDEO TRIMMING] Splitting videos.')
        for i in range(len(time_intervals)):
            print('[VIDEO TRIMMING] Creating video ' + vid_name + part_extensions[i])
            ffmpeg_extract_subclip(video_path, 0, time_intervals[i],
                                   targetname=os.path.join(video_dir, vid_name + part_extensions[i]))


def main():

    parser = argparse.ArgumentParser(description='Convert WeBots simulation snippets to video')
    parser.add_argument('--user', nargs='*', type=str, help='User for video perspective')
    parser.add_argument('--configuration', nargs='*', type=str, help='Object configuration')

    args = parser.parse_args()
    user = args.user[0]
    configuration = args.configuration[0]

    print('[VIDEO TRIMMING] Loading videos for configuration ' + configuration + ' and perspective for ' + user)
    video_dir = '../data/videos/configuration_' + configuration + '/' + user + '/'
    videos = [file for file in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, file))]
    part_extensions = ['_part_1.mp4', '_part_2.mp4', '_part_3.mp4']

    for video in videos:
        vid_full_idx = video.find('_full')
        if vid_full_idx != -1:
            vid_name = video[:vid_full_idx]
            split = True
            for part in part_extensions:
                if vid_name + part in videos:
                    split = False
                    break

            if split:
                print('[VIDEO TRIMMING] Splitting video ' + video)
                split_video(os.path.join(video_dir, video), video_dir, vid_name, [6, 12, 18], part_extensions)

            else:
                print('[VIDEO TRIMMING] Video ' + video + ' already splitted.')


if __name__ == '__main__':
    main()