# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Script to cut a video from the experiments footage
"""
import os.path as osp
from moviepy.editor import *

import pyrado
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt


# General settings
# moviesize = (1080, 1920)
font = 'LatinModernMath-Regular'  # Amiri-regular LatinModernMath-Regular
txt_color = 'white'
fontsize = 48
slide_duration = 4  # seconds
slide_duration_long = 7  # seconds

title_txt = """
Bayesian Domain Randomization (BayRn)
for Sim-to-Real Transfer


Fabio Muratore (1, 2),
Christian Eilers (1, 2),
Michael Gienger (2),
Jan Peters (1)

(1) Technical University Darmstadt, Germany
(2) Honda Research Institute Europe, Germany
"""

slide_wam_pre_txt = """
When learning from simulations, the optimizer is free to exploit the simulation.

Thus, the resulting policies can perform very well in simulation,
but transfer poorly to the real-world counterpart.

For example, both of the subsequent policies yield a return of 1,
and thus look equally good to the learner.
"""

slide_wam_inter_txt = """
Exemplary intermediate policies trained with
Bayesian Domain Randomization (BayRn)
"""

slide_wam_post_txt = """
BayRn uses a Gaussian process to learn how to adapt the randomized
simulator solely from the observed real-world returns.

BayRn is agnostic towards the policy optimization subroutine.
In this paper, we used PPO and PoWER.
"""

slide_qq_txt = """
We also evaluated BayRn on an underactuated swing-up and balance task. 
"""

if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    if not osp.isdir(args.dir):
        raise pyrado.PathErr(msg='Provide the path to the directory containing the videos using --dir')

    # Get the footage
    video_files_list = []
    wam_sim_f = VideoFileClip(osp.join(args.dir, 'WAM_MuJoCo_not_transferable.mp4'), audio=False)
    video_files_list.append(wam_sim_f)
    wam_sim_s = VideoFileClip(osp.join(args.dir, 'WAM_MuJoCo_transferable.mp4'), audio=False)
    video_files_list.append(wam_sim_s)
    wam_real_nom_f = VideoFileClip(osp.join(args.dir, 'WAM_fail.mp4'), audio=False)
    video_files_list.append(wam_real_nom_f)
    wam_real_bayrn_9 = VideoFileClip(osp.join(args.dir, 'WAM_BayRn_iter_9.mp4'), audio=False)
    video_files_list.append(wam_real_bayrn_9)
    wam_real_bayrn_4 = VideoFileClip(osp.join(args.dir, 'WAM_BayRn_iter_4.mp4'), audio=False)
    video_files_list.append(wam_real_bayrn_4)
    wam_real_bayrn_6 = VideoFileClip(osp.join(args.dir, 'WAM_BayRn_iter_6.mp4'), audio=False)
    video_files_list.append(wam_real_bayrn_6)
    qq_real = VideoFileClip(osp.join(args.dir, 'QQ_recover.mp4'), audio=False)
    video_files_list.append(qq_real)

    # Check the clips' sizes
    vid_w, vid_h = 0, 0
    if wam_sim_f.size == wam_sim_s.size == wam_real_nom_f.size == wam_real_bayrn_9.size == qq_real.size:
        print_cbt('Video sizes of the individual videos are equal.', 'g')
        vid_w, vid_h = wam_sim_f.size
    else:
        print_cbt('Video sizes of the individual videos are not equal.', 'y')
        for vid in video_files_list:
            print(f"{vid.filename[vid.filename.rfind('/'):]} size: {vid.size[0]} x {vid.size[1]}")
            vid_w = max(vid_w, vid.size[0]*2)  # *2 because the wam videos are put side by side
            vid_h = max(vid_h, vid.size[1])
        print_cbt(f'Video sizes inferred from the individual videos: {vid_w} x {vid_h}', 'w')

    title_intro = TextClip(title_txt, size=(vid_w, vid_h), font=font, fontsize=fontsize, color='white') \
        .set_position('center') \
        .set_start(0) \
        .set_duration(slide_duration) \
        .crossfadein(.3)

    # WAM
    wam_sim_array = clips_array([[wam_sim_f.set_duration(wam_sim_s.duration),
                                  wam_sim_s]]) \
        .resize(width=vid_w) \
        .set_start(slide_duration + slide_duration_long) \
        .crossfadein(.3)

    wam_real_array_1 = clips_array([[wam_real_nom_f, wam_real_bayrn_9.set_duration(wam_real_nom_f.duration)]]) \
        .margin(top=wam_sim_array.h - wam_real_nom_f.h) \
        .set_start(wam_sim_array.end) \
        .crossfadein(.3)

    wam_real_array_2 = clips_array([[wam_real_bayrn_4, wam_real_bayrn_6.set_duration(wam_real_bayrn_4.duration)]]) \
        .margin(top=wam_sim_array.h - wam_real_nom_f.h) \
        .set_start(wam_real_array_1.end + slide_duration) \
        .crossfadein(.3)

    txt_wam_left_1 = TextClip('Trained in the nominal simulation', font=font, color=txt_color, fontsize=fontsize,
                              align='West')
    txt_wam_left_1 = txt_wam_left_1. \
        set_duration(wam_sim_array.duration + wam_real_array_1.duration). \
        set_start(wam_sim_array.start). \
        set_position((vid_w*0.05, 0)). \
        crossfadein(.3)  # former pos y: txt_wam_left_1.h

    txt_wam_right_1 = TextClip('Trained with BayRn', font=font, color=txt_color, fontsize=fontsize, align='East')
    txt_wam_right_1 = txt_wam_right_1. \
        set_duration(wam_sim_array.duration + wam_real_array_1.duration). \
        set_start(wam_sim_array.start). \
        set_position((vid_w*0.95 - txt_wam_right_1.w, 0)). \
        crossfadein(.3)  # former pos y: txt_wam_right_1.h

    txt_wam_left_2 = TextClip('BayRn iteration 4', font=font, color=txt_color, fontsize=fontsize,
                              align='West')
    txt_wam_left_2 = txt_wam_left_2. \
        set_duration(wam_real_array_2.duration). \
        set_start(wam_real_array_2.start). \
        set_position((vid_w*0.05, 0)). \
        crossfadein(.3)  # former pos y: txt_wam_left_2.h

    txt_wam_right_2 = TextClip('BayRn iteration 6', font=font, color=txt_color, fontsize=fontsize,
                               align='East')
    txt_wam_right_2 = txt_wam_right_2. \
        set_duration(wam_real_array_2.duration). \
        set_start(wam_real_array_2.start). \
        set_position((vid_w*0.95 - txt_wam_right_2.w, 0)). \
        crossfadein(.3)  # former pos y: txt_wam_right_2.h

    slide_wam_pre = TextClip(slide_wam_pre_txt, font=font, fontsize=fontsize, color='white',
                             align='West') \
        .set_position('center') \
        .set_start(title_intro.end) \
        .set_duration(slide_duration_long) \
        .crossfadein(.3)

    slide_wam_inter = TextClip(slide_wam_inter_txt, font=font, fontsize=fontsize,
                               color='white', align='West') \
        .set_position('center') \
        .set_start(wam_real_array_1.end) \
        .set_duration(slide_duration) \
        .crossfadein(.3)

    slide_wam_post = TextClip(slide_wam_post_txt, font=font, fontsize=fontsize, color='white',
                              align='West') \
        .set_position('center') \
        .set_start(wam_real_array_2.end) \
        .set_duration(slide_duration_long) \
        .crossfadein(.3)

    # QQ
    slide_qq = TextClip(slide_qq_txt, font=font, fontsize=fontsize, color='white',
                        align='West') \
        .set_position('center') \
        .set_start(slide_wam_post.end) \
        .set_duration(slide_duration) \
        .crossfadein(.3)

    qq_real = qq_real.resize(height=vid_h) \
        .set_position("center") \
        .set_start(slide_qq.end) \
        .crossfadein(.3)

    txt_qq_left = TextClip('Trained with BayRn', font=font, color=txt_color, fontsize=fontsize, align='West')
    txt_qq_left = txt_qq_left. \
        set_start(qq_real.start). \
        set_duration(qq_real.duration). \
        set_position((vid_w*0.1, vid_h*0.4 + txt_qq_left.h)). \
        crossfadein(.3)

    txt_qq_right = TextClip('Applying disturbances', font=font, color=txt_color, fontsize=fontsize, align='East')
    txt_qq_right = txt_qq_right. \
        set_start(qq_real.start + qq_real.duration/3). \
        set_duration(qq_real.duration*2/3). \
        set_position((vid_w*0.9 - txt_qq_right.w, vid_h*0.4 + txt_qq_right.h)). \
        crossfadein(.3)

    title_outro = TextClip(title_txt, size=(vid_w, vid_h), font=font, fontsize=fontsize, color='white') \
        .set_position('center') \
        .set_start(qq_real.end) \
        .set_duration(slide_duration) \
        .crossfadein(.3)

    # Final assembly
    final = CompositeVideoClip([
        title_intro,
        slide_wam_pre, wam_sim_array, wam_real_array_1, wam_real_array_2, txt_wam_left_1, txt_wam_right_1,
        slide_wam_inter, txt_wam_left_2, txt_wam_right_2, slide_wam_post,
        slide_qq, qq_real, txt_qq_right,  txt_qq_left,
        title_outro
    ])
    final.write_videofile(osp.join(args.dir, 'BayRn_sim2real.mp4'), fps=30, codec='libx264', threads=8)
