"""
1.  Previous steps involve audio synthesis and script generation
    Script generation is kept manual but later on may be automated

2.  So in the movie format, we are going to make sure that the audio
    are generated at the sentence level only.

3.  Not a working solution for now. But giving an entire transcript to gpt and then asking for help
    works (kinda)

    As a hotfix creating interactive adder
    -> create a UI for improving UX
4. Input -> MovieGPTMiddleware object only
.get_context() will give dict
audio, data, movie clip information
and a movie

Pause the image on character introduction. Make Black and white and diff sub
Normal sub should be different
Transition between image clips
"""

import utility, defaults
import sys
import os
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from audio_generator import ThemeFactory

from moviepy.editor import TextClip, AudioClip, VideoClip, AudioFileClip, VideoFileClip, CompositeAudioClip, ImageSequenceClip, ImageClip, CompositeVideoClip
from moviepy.editor import concatenate_audioclips, concatenate_videoclips
import moviepy.video.fx.all as vfx
import random




def create_image_clips(subtitle_path):
    data = utility.read_json(subtitle_path)
    chapters = []
    for _data_ in data:
            cur_text = _data_['text']
            cur_start_time = _data_['start']
            cur_end_time = _data_['end']
            cur_chapter = _data_['chapter']
            if len(chapters) == 0 or chapters[-1][3] != cur_chapter:
                chapters.append([
                    cur_text,
                    cur_start_time,
                    cur_end_time,
                    cur_chapter
                ])
            else:
                last_e = chapters[-1]
                chapters[-1][0] += ' {}'.format(cur_text)
                chapters[-1][2] = cur_end_time

    image_clips = []        
    for chapter in chapters:
        prob = random.randint(0,2)
        if prob != 1:
            continue
        prompt = chapter[0]
        system = """You are a extremely wise stoic scene describer. Provide a vivid, highly detailed, photographic description of the following text
            Provide the final output in the following python dict. It should have a fields. Keep it within 200 words
            { "image_prompt" : ""
            }"""
        keys = set(['image_prompt'])
        prompt = utility.generate_dict_1(system, prompt, keys).get('image_prompt')
        negative_prompt = 'ugly'
        start_time = chapter[1]/1000
        end_time = chapter[2]/1000
        c_image = utility.get_hash() + '.jpeg'
        utility.generate_image_s(prompt, '', c_image)
        image_clip = (ImageClip(c_image)
                        .set_start(start_time)
                        .set_end(end_time)
                        .set_duration(end_time-start_time)
                        .crossfadein(0.5)
                        .crossfadeout(0.5)
                        .set_opacity(0.4))
        image_clips.append(image_clip)

    return image_clips 


def _make_(argument_map):

    workspace = argument_map.get('workspace')
    speech = argument_map.get('speech')

    workspace_folder = '/home/prakharrrr4/wolfy_ui/wolf/media/workspace/'
    movie_path = os.path.join(workspace_folder, 
                    os.path.join(workspace,'movie/movie.mp4'))
    video_clip = VideoFileClip(movie_path)

    driver = ThemeFactory().get_middleware(workspace,
                        theme = 'gpt_movie',
                        _id_ = speech)

    contexts = driver.get_internal_context()
    clips = []
    test = 3

    audio_clips = []
    video_clips = []

    for (chapter, rank, hash), snap in contexts.items():
        audio_file = snap['audio']
        audio_clip = AudioFileClip(audio_file)
        audio_clips.append(audio_clip)
        data_file = snap['data']
        data = utility.read_json(data_file)
        duration = audio_clip.duration

        movie_ts = data['movie_timestamps']
        movie_ts.sort()

        ranges = []

        for (left, right) in movie_ts:
            if len(ranges) == 0:
                ranges.append([left, right])
                continue
            else:
                if left <= ranges[-1][1]:
                    ranges[-1][1] = max(ranges[-1][1], right)
                else:
                    ranges.append([left,right])

        num = len(ranges)

        total = 0
        for l,r in ranges:
            total += r-l+1

        mapper = []
        rmapper = []
        weights = []
        s = 2

        for l,r in ranges:
            for k in range(l,r+1):
                rmapper.append(k)



        for i in range(total):
            mapper.append(i+1)
            weights.append((i+1)*s+2)
            s += 10
        weights.reverse()
        import random
        assert len(rmapper) == len(mapper)

        itr = 1000
        finalized = None
        while True:
            
            i = 0
            cur_list = []
            cur_dur = duration
            _bad = False
            while True:
                try:
                    j = random.choices(mapper[i:],weights[i:],k=1)[0]
                except:
                    j = len(mapper)-1
                k = j + random.randint(2,3)
                if k >= len(mapper):
                    _bad = True
                    cur_list.append([j,mapper[-1]])
                    break
                else:
                    min_v = min(cur_dur, k-j+1)
                    cur_dur -= min_v
                    cur_list.append([j,max(j,j+min_v-1)])
                    nxt = j+min_v-1
                    if nxt + 3 >= len(mapper):
                        _bad = True
                        break
                    else:
                        try:
                            i = random.choices(mapper[nxt+2:],weights[nxt+2:],k=1)[0]
                        except:
                            _bad = True
                            break
                if cur_dur == 0:
                    break

            if not _bad:
                finalized = cur_list
                break 

            itr -= 1
            if itr == 0:
                finalized = cur_list
                break
        
        cur = []

        for fin in finalized:
            l0 = rmapper[int(fin[0]-1)]
            r0 = rmapper[int(fin[1]-1)]+1
            cur.append(video_clip.subclip(l0,r0))

        final_clip = concatenate_videoclips(cur)
        final_clip = final_clip.fx(vfx.multiply_speed, final_duration = duration)
        final_clip = vfx.lum_contrast(final_clip, lum = 5, contrast = 0.2)
        # final_clip = utility.warmth(final_clip)
        final_clip = utility.invert(final_clip)
        final_clip.audio = None
        final_clip.audio = audio_clip
        clips.append(final_clip)

    clips = concatenate_videoclips(clips)
    audio_clip = clips.audio
    c_credit = utility.get_hash() + '.json'
    c_audio_bg = utility.get_hash() + '.mp3'
    utility.add_background_music(audio_clip, result_file = c_audio_bg, credit_file = c_credit)
    audio_clip = AudioFileClip(c_audio_bg)
    final_clip.audio = audio_clip

    clips.write_videofile('result.mp4')





def run_interaction(args):
    
    movie_gpt = ThemeFactory().get_middleware(
        args.get('workspace'),
        theme = 'gpt_movie',
        _id_ = args.get('speech'),
    )

    context = movie_gpt.get_internal_context()
    for (chapter, rank, hash), value in context.items():
        data_file = value.get('data')
        data = utility.read_json(data_file)
        if data.get('movie_timestamps',None) is not None:
            continue
        data['movie_timestamps'] = list()
        print("**************SPEEECH***************\n")
        print(data.get('speech'))
        print("**************SPEEECH***************\n")
        while True:
            st = int(input('ENTER START [-1 to exit current speech] : '))
            if st == -1:
                break
            en = int(input('ENTER END : '))
            data['movie_timestamps'].append([st,en])
        utility.save_json(data, data_file)


if __name__ == '__main__':

    temp_fix = sys.argv[1]
    args = {
        'workspace' : 'a',
        'speech' : 'movie1',
    }
    if temp_fix == 'y':
        run_interaction(args)
    if temp_fix == 'm':
        _make_(args)




