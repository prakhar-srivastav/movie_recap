"""
1.  Previous steps involve audio synthesis and script generation
    Script generation is kept manual but later on may be automated

2.  So in the movie format, we are going to make sure that the audio
    are generated at the sentence level only.

3.  Not a working solution for now. But giving an entire transcript to gpt and then asking for help
    works (kinda) It is research item titled movie vector database

4.  Where as known solution : As a hotfix create interactive UI to recommend the clips
    and allow suggestion much like a match recommender. While try giving some suggestion
    via clip or blip or try fine tuning. This research item should be prioratized voer 
    preset_3

5. Input -> MovieGPTMiddleware object only
    .get_context() will give dict
    audio, data, movie clip information
    and a movie

6. Pause the image on character introduction. Make Black and white and diff sub
    Normal sub should be different

7. Transition between image clips
"""
import sys
import os

from . import utility, defaults

import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from audio_generator import ThemeFactory

from moviepy.editor import TextClip, AudioClip, VideoClip, AudioFileClip, VideoFileClip, CompositeAudioClip, ImageSequenceClip, ImageClip, CompositeVideoClip
from moviepy.editor import concatenate_audioclips, concatenate_videoclips
import moviepy.video.fx.all as vfx
import moviepy.audio.fx.all as afx
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


def add_text_to_clip(argument_map):
    workspace = argument_map['workspace']
    instance = argument_map['instance']
    character_information = argument_map['character_information']
    clips = []
    workspace_folder = '/home/prakharrrr4/movie_recap/media/workspace/'
    movie_path = os.path.join(workspace_folder, 
                    os.path.join(workspace,'movie/result.mp4'))
    ret_path = os.path.join(workspace_folder, 
                    os.path.join(workspace,'movie/result_with_txt.mp4'))
    video_clip = VideoFileClip(movie_path)
    clips.append(video_clip)

    for c1 in character_information:
        time, char, pos = c1
        text = char
        delta = 1.5
        start_time = time 
        end_time = time + delta
        position_of_text = (0.3,0.5)
        

        txt_clip = (TextClip(text, font=defaults.preset_2_ttf,
                    font_size=120, color='white', size=(1200, None),
                    kerning = 1, method='caption')
                            .with_position(position_of_text, relative = True)
                            .set_start(start_time)
                            .set_end(end_time)
                            .set_duration(end_time-start_time)
                            .crossfadein(0.5) 
                            .crossfadeout(0.5))
        clips.append(txt_clip)

    final_video = CompositeVideoClip(clips)
    final_video.write_videofile(ret_path,
                        fps = 24,
                        threads = 16,
                        preset='ultrafast')
    return ret_path


def create_yt_clip(argument_map):

    print("Creating YT Clip")
    workspace = argument_map.get('workspace',None)
    speech = argument_map.get('instance',None)
    workspace_folder = '/home/prakharrrr4/movie_recap/media/workspace/'
    yt_video_path = os.path.join(workspace_folder, 
                    os.path.join(workspace,'movie/yt_video.mp4'))
    video_clip = VideoFileClip(yt_video_path)

    driver = ThemeFactory().get_middleware(workspace,
                        theme = 'gpt_movie',
                        _id_ = speech)

    contexts = driver.get_internal_context()

    video_clips = []
    lim = 1
    result_file = None
    for (chapter, rank, hash), snap in contexts.items():
        
        print("Chapter : {}, Rank : {}".format(chapter, rank))
        key = str(chapter) + '_' + str(rank) + '_' + hash
        
        data_file = snap['data']
        data = utility.read_json(data_file)
        audio_file = snap['audio']
        audio_clip = AudioFileClip(audio_file)
        target_duration = audio_clip.duration
        duration = data['duration']
        start = data['start']
        timestamp = []
        start = int(start)
        duration = int(duration)

        final_clip = video_clip.subclip(start,start+duration)
        final_clip = vfx.multiply_speed(final_clip, final_duration = target_duration)
        final_clip.audio = None
        video_clips.append(final_clip)
        

    clip = concatenate_videoclips(video_clips)
    result_file = os.path.join(workspace_folder, 
                    os.path.join(workspace,'movie/result_yt.mp4'))
    clip.write_videofile(result_file)
    return result_file


def create_clip(argument_map):

    print("Creating Clip")
    workspace = argument_map.get('workspace',None)
    speech = argument_map.get('instance',None)
    _id_ = argument_map.get('_ids_',None)
    timestamp = argument_map.get('timestamp',None)
    workspace_folder = '/home/prakharrrr4/movie_recap/media/workspace/'
    movie_path = os.path.join(workspace_folder, 
                    os.path.join(workspace,'movie/movie.mp4'))
    video_clip = VideoFileClip(movie_path)

    driver = ThemeFactory().get_middleware(workspace,
                        theme = 'gpt_movie',
                        _id_ = speech)

    contexts = driver.get_internal_context()

    video_clips = []
    lim = 1
    result_file = None
    for (chapter, rank, hash), snap in contexts.items():
        
        print("Chapter : {}, Rank : {}".format(chapter, rank))
        key = str(chapter) + '_' + str(rank) + '_' + hash
        if timestamp is not None and key != _id_:
            continue
        
        audio_file = snap['audio']
        audio_clip = AudioFileClip(audio_file)
        data_file = snap['data']
        data = utility.read_json(data_file)
        duration = audio_clip.duration
        
        if timestamp is not None:
            data['movie_timestamps'] = timestamp
            utility.save_json(data, data_file)

        try:
            movie_ts = data['movie_timestamps']            
        except:
            import pdb; pdb.set_trace()
        cur = []
        vis = set()
        for ts in movie_ts:
            if ts not in vis:
                cur.append(video_clip.subclip(ts,ts+lim))
                vis.add(ts)
            else:
                cur.append(None)
        
        last_good = -1
        for i in range(len(cur)):
            if cur[i] is None:
                continue
            else:
                if i - last_good > 1:
                    cur[last_good + 1] = vfx.freeze(cur[last_good], t = "end", freeze_duration = i - last_good - 1)
                last_good = i
        
        new_cur = []

        for entry in cur:
            if entry is not None:
                new_cur.append(entry)

        cur = new_cur
        # new_cur = []
        # for c1 in cur:
        #     c1 = vfx.fadein(c1, duration = 0.3)
        #     c1 = vfx.fadeout(c1, duration = 0.3)
        #     new_cur.append(c1)
        # cur = new_cur
        final_clip = concatenate_videoclips(cur)
        final_clip = vfx.lum_contrast(final_clip, lum = -3, contrast = 0.1)
        final_clip = vfx.multiply_speed(final_clip, final_duration = duration)
        # final_clip = utility.warmth(final_clip)
        # final_clip = utility.invert(final_clip)
        final_clip.audio = None
        final_clip.audio = audio_clip
        video_clips.append(final_clip)
        result_file = snap['data'].replace('data.json','clip.mp4')
        
    clip = concatenate_videoclips(video_clips)
    c_credit = utility.get_hash() + '.json'
    c_audio_bg = utility.get_hash() + '.mp3'
    # utility.add_background_music(clip.audio, c_audio_bg, defaults.quotes_specific, factor = 0.4,credit_file = c_credit)
    # audio_clip = AudioFileClip(c_audio_bg)
    audio_clip = clip.audio
    audio_clip = audio_clip.fx(afx.multiply_volume, factor = 4.0)
    clip.audio = audio_clip
    if timestamp is None:
        result_file = os.path.join(workspace_folder, 
                    os.path.join(workspace,'movie/result.mp4'))
    clip.fps = 21
    clip.write_videofile(result_file)
    clip.write_videofile("output_video.mp4", preset='fast')

    return result_file


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


    # import pdb; pdb.set_trace()

# if __name__ == '__main__':
#     # test
#     args = {
#         'workspace' : 'a',
#         'speech' : 'skibdi_story',
#         'hash' : '5d6dd727b69b4882bc2c3ae9ad591ab1',
#         'timestamps' : [0,4,6,8,10]
#     }
#     create_clip(args)
#     # _make_(args)
#     # if temp_fix == 'y':
#     #     run_interaction(args)
#     # if temp_fix == 'm':
#     #     _make_(args)




