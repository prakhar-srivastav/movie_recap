import uuid
import os
import threading
import time
from TTS.api import TTS
import whisper_timestamped as whisper
import uuid
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from concurrent.futures import ProcessPoolExecutor
from transformers import CLIPProcessor, CLIPModel
from youtube_transcript_api import YouTubeTranscriptApi
from pydub.playback import play
from jiwer import wer
import json
import sys
from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageEnhance
import glob
import torch
import pickle
import zipfile
from tqdm.autonotebook import tqdm
torch.set_num_threads(4)


"""
T E S T
"""
# sys.path.append('/home/prakharrrr4/movie_recap/wolf/wolf')
# import setting2 as settings
# import scraper_middleware as sm

"""
P R O D
"""
from django.conf import settings
import workspace.scraper_middleware as sm



"""
AudioAudit :
- synthesize function :: It will create the audio data and it will keep the timing of the genrated audio file to later on correct

audio_audit_object.synthesize(text : str = None, speaker : str = None) -> compiles speaker wav output and internally prepare timings of the wav
                                            loads the audio and time mappings
    audio_audit_object.feedback(timings : list = []) -> Takes the timings of the faulty audios and refinds the incorrect audiofiles and regenerates along 
                                            with modified timings. 
                                            modifies the audio and time mappings
    audio_audit.save_file(result_path : str = None)    
"""

class AudioGenerator(object):

    def __init__(self):
        self.timings = dict()
        self.combined_audio = None
        self._id_ = uuid.uuid4().hex
        self.tts =  TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        self.PATH = '/home/prakharrrr4/pegasus/data/tmp'


    def _sentence_splitter_(
                    self,
                    text,
                    ):
        from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
        import torch
        import pysbd

        vocab_file = "/home/prakharrrr4/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/vocab.json"
        tokenizer = VoiceBpeTokenizer(vocab_file)

        def get_token(sent):
            return torch.IntTensor(tokenizer.encode(sent, lang=language)).unsqueeze(0).to("cuda").shape[-1]
        
        limit = 300
        segmenter = pysbd.Segmenter(language="en", clean=False)
        sentences = segmenter.segment(text)
        result = [sentences[0]]
        for i in range(1,len(sentences)):
            sen = sentences[i]
            _trial_ = (" ").join([result[-1],sen])
            sen_tok = get_token(_trial_)
            if sen_tok <= limit:
                result[-1] = _trial_
            else:
                result.append(sen)
        return result



    def prepare_data_from_audio_files_and_texts(self, sentences, audio_files):
        from pydub import AudioSegment
        from pydub.playback import play

        time_mappings=  {}
        combined_audio = AudioSegment.empty()
        
        last_time = 0
        for i in range(len(audio_files)):
            file = audio_files[i]
            sen = sentences[i]
            current_audio0 = AudioSegment.from_file(file)
            current_audio = current_audio0 + AudioSegment.silent(duration=250)
            new_last_time = last_time +  len(current_audio)
            time_mappings[(last_time, new_last_time)] = (current_audio0, sen)
            combined_audio += current_audio
            last_time = new_last_time

        return time_mappings, combined_audio


    def synthesize(self, text = '', speaker = '', output = ''):

        self.speaker = speaker
        _folder_hash_ = uuid.uuid4().hex
        _hash_ = uuid.uuid4().hex + '.wav'

        sentences = self._sentence_splitter_(text)
        audio_files = []
        _folder_path_ = os.path.join(self.PATH, _folder_hash_)

        os.makedirs(_folder_path_, exist_ok = True)

        itr = 0

        for sen in sentences:
            
            path = os.path.join(_folder_path_,_folder_hash_ + '_{}.wav'.format(itr))
            self.tts.tts_to_file(text=sen,  
                            file_path= path,
                            speaker_wav=[speaker],
                            language="en",
                            split_sentences=False 
                            )
            audio_files.append(path)
            itr += 1

        self.timings, self.combined_audio = self.prepare_data_from_audio_files_and_texts(sentences, audio_files)


    def feedback(self, faulty_timings, is_seconds = False):
    
        if is_seconds:
            faulty_timings = [time*1000 for time in faulty_timings]

    
        faulty_timings.sort()      
        bad_keys = set()
        for f in faulty_timings:
            for (s,e), data in self.timings.items():
                if f>=s and f<=e:
                    bad_keys.add((s,e))
        
        _folder_hash_ = uuid.uuid4().hex
        _folder_path_ = os.path.join(self.PATH, _folder_hash_)
        os.makedirs(_folder_path_, exist_ok = True)
        itr = 0

        audio_files = []
        sentences = []

        for key, value in self.timings.items():
            cur_audio, sen = value
            path = os.path.join(_folder_path_,_folder_hash_ + '_{}.wav'.format(itr))
            if key in bad_keys:
                print(itr,'tts')
                self.tts.tts_to_file(text=sen,  
                                file_path= path,
                                speaker_wav=[self.speaker],
                                language="en",
                                split_sentences=False 
                                )
            else:
                print(itr,'audio')
                cur_audio.export(path, format="wav")
            audio_files.append(path)
            sentences.append(sen)
            itr += 1

        self.timings, self.combined_audio = self.prepare_data_from_audio_files_and_texts(sentences, audio_files)


    def save_file(self, result_path):
        self.combined_audio.export(result_path, format="wav")    


class ThemeFactory(object):

    def __init__(self):
        self._theme_  = ['xtts', 'gpt_quote', 'gpt_movie', 'gpt_recap']

    def get_middleware(self,workspace, theme = 'gpt_recap', _id_ = None):
        
        theme = theme.strip()
        if theme == 'xtts' or theme == '':
            return WolfyTTSMiddleware(workspace, _id_ = _id_)
        elif theme == 'gpt_quote':
            return QuotesGPTMiddleware(workspace, _id_ = _id_)
        elif theme == 'gpt_movie':
            return MovieGPTMiddleware(workspace, _id_ = _id_)
        elif theme == 'gpt_recap':
            return RecapGPTMiddleware(workspace, _id_ = _id_)
        else:
            raise ValueError("Not yet implemented")


class WolfyTTSMiddleware(object):

    def __init__(self, workspace, _id_ : str = None):
        self.workspace = workspace
        self.PATH = os.path.join(settings.MEDIA_ROOT,'workspace')
        self._id_ = _id_ if _id_ else uuid.uuid4().hex
        os.makedirs(self.get_folder_path(),exist_ok = True)


    def _sentence_splitter_simple_(
                    self,
                    text,
    ):
        import pysbd
        segmenter = pysbd.Segmenter(language="en", clean=False)
        sentences = segmenter.segment(text)
        return sentences


    def _sentence_splitter_(
                    self,
                    text,
                    ):
        from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
        import torch
        import pysbd

        vocab_file = "/home/prakharrrr4/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/vocab.json"
        tokenizer = VoiceBpeTokenizer(vocab_file)

        def get_token(sent):
            return torch.IntTensor(tokenizer.encode(sent, lang="en")).unsqueeze(0).to("cuda").shape[-1]
        
        limit = 130
        segmenter = pysbd.Segmenter(language="en", clean=False)
        sentences = segmenter.segment(text)
        result = [sentences[0]]
        for i in range(1,len(sentences)):
            sen = sentences[i]
            _trial_ = (" ").join([result[-1],sen])
            sen_tok = get_token(_trial_)
            if sen_tok <= limit:
                result[-1] = _trial_
            else:
                result.append(sen)
        print("CODENAME")
        print(result)
        for sen in result:
            print(get_token(sen))
        return result


    def read_json(self, filename):
        try:
            j = open(filename, 'r')
            json_object =  json.loads(j.read())
            j.close()
            return json_object
        except:
            return {}

    def save_json(self, request_body, filename):
        json_object = json.dumps(request_body, indent=4)    
        with open(filename, "w") as outfile:
            outfile.write(json_object)


    def save_with_silence(self, text="", file_path = "", speaker_wav = "", silence = 250):
        if text == "":
            current_audio = AudioSegment.silent(duration=silence)
        else:
            
            # if not hasattr(self,'tts'):
            #     self.tts = TTS('tts_models/multilingual/multi-dataset/bark', gpu = True)

            # self.tts.tts_to_file(text=text,
            #     file_path=file_path,
            #     voice_dir="bark_voices/",
            #     speaker="v2/en_speaker_6")

            if not hasattr(self, 'tts'):
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
            self.tts.tts_to_file(text=text,  
                                    file_path= file_path,
                                    speaker_wav= speaker_wav,
                                    language="en",
                                    split_sentences=False 
                                    )

            current_audio0 = AudioSegment.from_file(file_path)
            current_audio = current_audio0 + AudioSegment.silent(duration=silence)
        time = len(current_audio)
        current_audio.export(file_path, format="wav") 
        return time


    def save_audio_instance(
                            self,
                            text : str = None,
                            file_path : str = None,
                            speaker_wav : str = None,
                            silence : float = None,
                            rank : int = None,
                            chapter : int = None,
                            ):

            time = self.save_with_silence(text=text, file_path= file_path, speaker_wav=speaker_wav, silence = silence)
            asr_speech, wer, segment_timestamps = self.get_asr_and_wer(text, file_path)
            data = {
                'time' : time,
                'speaker' : speaker_wav,
                'rank' : rank,
                'chapter' : chapter,
                'speech' : text,
                'asr_speech' : asr_speech,
                'wer' : wer,
                'silence' : silence,
                'segment_timestamps' : segment_timestamps
            }
            data_path = file_path.replace("speech.wav","data.json")
            self.save_json(data, data_path)


    def get_asr_and_wer(self, text, audio_file):
        try:
            if text == "":
                return "" , 0.0, None
            if not hasattr(self, 'asr_model'):
                self.asr_model = whisper.load_model("base")
            audio = whisper.load_audio(audio_file)
            result = whisper.transcribe(self.asr_model, audio)        
            return result['text'], wer(text, result['text']), result
        except:
            return "" , 0.0, None


    def get_folder_path(self):
        return  os.path.join(self.PATH,
                os.path.join(self.workspace,
                os.path.join('audio',self._id_)))


    def get_main_folder_path(self):
        return  os.path.join(self.PATH,
                os.path.join(self.workspace))


    def synthesize(self, text = '', speaker = '', youtube_link = '', output = ''):
        speaker = self.get_speaker_path(speaker)

        _folder_path_ = self.get_folder_path()
        sentences = self._sentence_splitter_(text)
        audio_files = []

        os.makedirs(_folder_path_, exist_ok = True)

        for itr in range(len(sentences)):
            sen = sentences[itr]
            _hash_ = uuid.uuid4().hex
            audio_folder = os.path.join(_folder_path_, _hash_)
            os.mkdir(audio_folder)
            audio_file = os.path.join(audio_folder, 'speech.wav')
            self.save_audio_instance(text = sen,
                                     file_path = audio_file,
                                     speaker_wav = speaker,
                                     silence = 250,
                                     rank = itr + 1,
                                     chapter = 1,
                                     )


    def get_speaker_path(self, speaker_name):
        PATH = os.path.join(settings.MEDIA_ROOT,'data/audio_data')
        return PATH + '/' + speaker_name + '.wav' 


    def filesystem_to_media_url(self, absolute_file_path):
        media_root = os.path.normpath(settings.MEDIA_ROOT)
        absolute_file_path = os.path.normpath(absolute_file_path)
        
        if absolute_file_path.startswith(media_root):
            relative_path = absolute_file_path[len(media_root):].lstrip(os.path.sep)
            media_url = os.path.join(settings.MEDIA_URL, relative_path).replace('\\', '/')
            return media_url
        else:
            raise ValueError("The provided file path is not under MEDIA_ROOT.")


    def get_context(self, _ids_=None):
        folder_path = self.get_folder_path()
        audio_files = os.listdir(folder_path)
        
        context = dict()
        for audio_file in audio_files:
            if _ids_ and audio_file not in _ids_:
                continue
            audio_path = os.path.join(folder_path,audio_file)
            data = self.read_json(os.path.join(audio_path, 'data.json'))
            speech_wav = os.path.join(audio_path, 'speech.wav')
            key = '{}_{}_{}'.format(
                str(data['chapter']),
                str(data['rank']),
                str(audio_file)
            )
            video_url = os.path.join(audio_path, 'clip.mp4')
            if not os.path.exists(video_url):
                video_url = None
            context[key] = {
                                'time' : data['time']/1000,
                                'rank' : data['rank'],
                                'chapter' : data['chapter'],
                                'speaker' : data['speaker'].split('/')[-1].split('.')[0],
                                'speech' : data['speech'],
                                'asr_speech' : data['asr_speech'],
                                'wer' : data['wer'],
                                'audio_url' : self.filesystem_to_media_url(speech_wav),
                                'hash' : audio_file,
                                'video_url' : self.filesystem_to_media_url(video_url) if video_url else None,
                                'all' : data
                            }
            
        def sort_context(context):
            context_array = []
            for key, value in context.items():
                chapter, rank, _hash_ = key.split('_')
                chapter = int(chapter)
                rank = int(rank)
                context_array.append((chapter, rank, _hash_, key, value))
            context_array.sort()
            context = {}
            for (chapter, rank, _hash_, key, value) in context_array:
                context[key] = value
            return context
        context = sort_context(context)
        return context


    def get_instance(self):
        return self._id_


    def get_movie_url(self):
        folder_path = self.get_folder_path()
        parent_folder = os.path.dirname(os.path.dirname(folder_path))
        movie_url = os.path.join(parent_folder, 'movie/movie.mp4')
        return movie_url


    def get_result_video_url(self):
        folder_path = self.get_folder_path()
        parent_folder = os.path.dirname(os.path.dirname(folder_path))
        movie_url = os.path.join(parent_folder, 'movie/result.mp4')
        return movie_url


    def get_result_yt_video_url(self):
        folder_path = self.get_folder_path()
        parent_folder = os.path.dirname(os.path.dirname(folder_path))
        movie_url = os.path.join(parent_folder, 'movie/result_yt.mp4')
        return movie_url


    def save_file(self, nickname):
        older_folder_path = self.get_folder_path()
        self._id_ = nickname
        new_folder_path = self.get_folder_path()
        os.rename(older_folder_path, new_folder_path)


    def delete_by_ids(self, _ids_):
        if type(_ids_) != list:
            _ids_ = [_ids_]

        _ids_ = set(_ids_)
        _folder_path_ = self.get_folder_path()

        cur_c = 0
        cur_r = 1

        def update_record(hash, c, r):
            audio_folder = os.path.join(_folder_path_, hash)
            audio_data_path = os.path.join(audio_folder, 'data.json')
            data = self.read_json(audio_data_path)
            data['chapter'] = c
            data['rank'] = r
            self.save_json(data, audio_data_path)

        
        def delete_record(hash):
            audio_folder = os.path.join(_folder_path_, hash)
            import shutil
            shutil.rmtree(audio_folder)


        for key, value in self.get_internal_context().items():
            con_key = ('_').join([str(x) for x in key])
            c, r, hash = key
            if con_key not in _ids_:
                if c != cur_c:
                    cur_c += 1
                    cur_r = 1
                update_record(hash, cur_c, cur_r)
                cur_r += 1
            else:
                delete_record(hash)


    def regenerate_by_ids(self, _ids_):
        if type(_ids_) != list:
            _ids_ = [_ids_]
        
        _folder_path_ = self.get_folder_path()

        _new_ids_ = []
        for _id_ in _ids_:
            _new_ids_.append(_id_.split('_')[-1])
        _ids_ = _new_ids_
        for _id_ in _ids_:
            
            audio_folder = os.path.join(_folder_path_, _id_)
            audio_data_path = os.path.join(audio_folder, 'data.json')
            audio_file = os.path.join(audio_folder, 'speech.wav')
            data = self.read_json(audio_data_path)
            self.save_audio_instance(text = data.get('speech'),
                            file_path = audio_file,
                            speaker_wav = data.get('speaker'),
                            silence = data.get('silence'),
                            rank = data.get('rank'),
                            chapter = data.get('chapter'),
                            )
        return self.get_context(set(_ids_))


    def get_internal_context(self):
        folder_path = self.get_folder_path()
        audio_files = os.listdir(folder_path)
        
        context = dict()
        for audio_file in audio_files:
            audio_path = os.path.join(folder_path,audio_file)
            data = self.read_json(os.path.join(audio_path, 'data.json'))
            speech_wav = os.path.join(audio_path, 'speech.wav')
            try:
                key = (data['chapter'], data['rank'], audio_file)
            except:
                continue
            context[key] = {
                                'rank' : data['rank'],
                                'chapter' : data['chapter'],
                                'data' : os.path.join(audio_path, 'data.json'),
                                'audio' : os.path.join(audio_path, 'speech.wav'),
                                'audio_path' : audio_path,
                                'hash' : audio_file,
                                'time' : data['time']
                            }
            
        def sort_context(context):
            context_array = []
            for key, value in context.items():
                (chapter, rank, _hash_) = key
                chapter = int(chapter)
                rank = int(rank)
                context_array.append((chapter, rank, _hash_, key, value))
            context_array.sort()
            context = {}
            for (chapter, rank, _hash_, key, value) in context_array:
                context[key] = value
            return context
        context = sort_context(context)
        return context


    def __iadd__(self, other):
        _a_ = self.get_internal_context()
        _b_ = other.get_internal_context()
        max_chapter = 0
        for (chapter,_,_) in _a_.keys():
            max_chapter = max(max_chapter, chapter)
        
        import shutil
        _a_folder = self.get_folder_path()
        _b_folder = other.get_folder_path()
        for (_,_,_),values in _b_.items():
            audio_path = values['audio_path']
            shutil.copytree(audio_path, _a_folder + '/{}'.format(values['hash']), dirs_exist_ok = True)
            new_path = os.path.join(_a_folder, values['hash'])
            data_path = os.path.join(new_path, 'data.json')
            speech_path = os.path.join(new_path, 'speech.wav')
            data = self.read_json(data_path)
            data['chapter'] += max_chapter
            self.save_json(data, data_path)
        return self


    def get_total_time(self):
        context = self.get_internal_context()
        time = 0
        for key,value in context.items():  
            time += value['time']
        return time


    def add_silence_audio(self, silence = 250):
        context = self.get_internal_context()
        max_c = 1
        max_r = 1
        for c,r,_ in context.keys():
            max_c = max(c,max_c)
        for c,r,_ in context.keys():
            if c == max_c:
                max_r = max(max_r, r) 

        max_r += 1
        file_path = os.path.join(self.get_folder_path(),uuid.uuid4().hex)
        os.mkdir(file_path)
        audio_path = os.path.join(file_path, 'speech.wav')
        self.save_audio_instance(
                            "",
                            audio_path,
                            "",
                            silence,
                            max_r,
                            max_c
                            )


    def get_subtitles(self):
        context = self.get_internal_context()
        subs = []
        timer = 0
        for key, value in context.items():
            data = self.read_json(value['data'])
            if 'segment_timestamps' in data.keys() and data['segment_timestamps'] is not None:
                for segment in data['segment_timestamps'].get('segments',[]):
                    
                    words = []

                    for itr in segment['words']:
                        words.append({
                            'start' : timer + itr['start'] * 1000,
                            'end' : timer + itr['end'] * 1000,
                            'text' : itr['text']
                        })

                    subs.append({
                        'start' : timer + segment['start'] * 1000,
                        'end' : timer + segment['end'] * 1000,
                        'text' : segment['text'],
                        'chapter' : value['chapter'],
                        'words' : words,
                        'rank' : value['rank']
                    })
            timer += value['time']

        return subs


    def get_subtitles_at_word_level(self):
        words = []

        for sen in self.get_subtitles():
            for word in sen['words']:
                words.append({
                    'start' : word['start'],
                    'end' : word['end'],
                    'text' : word['text'],
                    'chapter' : sen['chapter'],
                    'rank' : sen['rank'],
                })

        return words

class QuotesGPTMiddleware(WolfyTTSMiddleware):

    def __init__(self, workspace, _id_ = None):
        super().__init__(workspace, _id_ = _id_)
        pass


    def synthesize(self, text = '', speaker = '', youtube_link = '', output = ''):

        speaker = 'gpt_cove'
        _folder_path_ = self.get_folder_path()

        author_text = text.split('\n')
        author_list = [x.strip() for x in author_text]

        sentences = []

        for author in author_list:
            cur = sm.extract_quotes(author)
            sentences.extend(cur)
        audio_files = []

        os.makedirs(_folder_path_, exist_ok = True)

        for itr in range(len(sentences)):
            sen = sentences[itr]
            _hash_ = uuid.uuid4().hex
            audio_folder = os.path.join(_folder_path_, _hash_)
            os.mkdir(audio_folder)
            audio_file = os.path.join(audio_folder, 'speech.wav')
            silence = 3000
            self.save_audio_instance(text = sen,
                                     file_path = audio_file,
                                     speaker_wav = speaker,
                                     silence = silence,
                                     rank = itr + 1,
                                     chapter = 1,
                                     )


    def save_with_silence(self, text="", file_path = "", speaker_wav = "", silence = 250):
        if text == "":
            current_audio = AudioSegment.silent(duration=silence)
        else:
            sm.extract_audio_from_chat_gpt(text, file_path)
            sm.alter_audio(file_path, affects = 'slowed-reverb')
            current_audio0 = AudioSegment.from_file(file_path)
            current_audio = current_audio0 + AudioSegment.silent(duration=silence)
        time = len(current_audio)
        current_audio.export(file_path, format="wav") 
        return time


class MovieGPTMiddleware(WolfyTTSMiddleware):

    def __init__(self, workspace, _id_ = None):
        super().__init__(workspace, _id_ = _id_)
        pass


    def synthesize(self, text = '', speaker = '', youtube_link = '', output = ''):

        speaker = 'gpt_cove'
        _folder_path_ = self.get_folder_path()

        audio_files = []
        

        sentences = self._sentence_splitter_(text)
        os.makedirs(_folder_path_, exist_ok = True)

        for itr in range(len(sentences)):
            sen = sentences[itr]
            _hash_ = uuid.uuid4().hex
            audio_folder = os.path.join(_folder_path_, _hash_)
            os.mkdir(audio_folder)
            audio_file = os.path.join(audio_folder, 'speech.wav')
            self.save_audio_instance(text = sen,
                                     file_path = audio_file,
                                     speaker_wav = speaker,
                                     silence = 1,
                                     rank = itr + 1,
                                     chapter = 1,
                                     )


    def save_with_silence(self, text="", file_path = "", speaker_wav = "", silence = 250):
        if text == "":
            current_audio = AudioSegment.silent(duration=silence)
        else:
            sm.extract_audio_from_chat_gpt(text, file_path)
            current_audio0 = AudioSegment.from_file(file_path)
            current_audio = current_audio0 + AudioSegment.silent(duration=silence)
        time = len(current_audio)
        current_audio.export(file_path, format="wav") 
        return time


def save_frame(args):
    video_path, i_s, output_dir, worker_id = args
    print("worker_id : {} START".format(worker_id))
    clip = VideoFileClip(video_path)
    for i in i_s:
        frame_path = os.path.join(output_dir, f"{i}.png")
        clip.save_frame(frame_path, t=i)
    print("worker_id : {} DONE".format(worker_id))


system = '''User will give you are paragraph. This GPT job is to transform it giving the tone of human. Example of human tone (DONT USE ANY FACTUAL INFORMATION FROM THE EXAMPLE.  JUST USE THE SPEAKING STYLE). Keep the sentences short and precise. Don't elongate the answer. Keep number of words same
```
In mid-1920s Copenhagen, portrait artist Gerda Wegener asks her husband, a popular landscape artist and closeted trans woman (then going by the name Einar Wegener), to stand in for a female model, who is late arriving at their flat to pose for a painting Gerda is working on.
```

KEEP IT SIMPLE
DONT ACT SMART

DONT START WITH 'In mid-1920s Copenhagen,'
DONT USE ANY COMPLEX ENGLISH
WRITE IN ESSAY FORMAT LIKE A 8th STANDARD CHILD WRITING

Provide the final output in the following python dict. It should have a fields. Keep it within 200 words
            { "data
" : ""
            }'''


class RecapGPTMiddleware(WolfyTTSMiddleware):


    def __init__(self, workspace, _id_ = None):
        super().__init__(workspace, _id_ = _id_)
        pass


    def download_yt(self, link, output_path):
        try:
            from pytube import YouTube
            yt = YouTube(link)
            stream = yt.streams.first()
            stream.download(filename = output_path)
        except:
            import yt_dlp
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': output_path,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([link])


    def divide_chunks(self, lst, n):
        chunk_size = len(lst) // n
        chunks = [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
        remainder = len(lst) % n
        if remainder:
            for i in range(1, remainder + 1):
                chunks[-i].append(lst[-i])
        
        return chunks


    def create_image(self, video_path, output_dir, hidden_ranges = []):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(output_dir)
        json_file = os.path.join(output_dir,'data.json')
        image_folder = os.path.join(output_dir,'images')
        data = self.read_json(json_file)
        if data.get('video_path') == video_path:
            data['hidden_ranges'] = hidden_ranges
            print("Skipping")
            return 
        try:
            shutil.rmtree(image_folder)
        except: 
            pass
        data['video_path'] = video_path
        data['hidden_ranges'] = hidden_ranges
        self.save_json(data,json_file)

        os.makedirs(image_folder, exist_ok = True)

        duration = int(VideoFileClip(video_path).duration)
        print(duration)
        num_workers = 8
        inps = [i for i in range(0, duration, 1)]
        inps = self.divide_chunks(inps, num_workers)
        inps = [(video_path, x, image_folder) for x in inps]
        new_inps = []
        for i in range(len(inps)):
            inp = inps[i]
            new_inp = (inp[0], inp[1], inp[2], str(i))
            new_inps.append(new_inp)
        inps = new_inps
        with ProcessPoolExecutor() as executor:
            list(executor.map(save_frame, inps))


    def compute_embedding(self, image_folder, serialize = False):
        image_folder = os.path.join(image_folder, 'images')
        st = time.time()
        print('starting to generate image embedding')
        model = SentenceTransformer('clip-ViT-B-32')
        image_list = self.get_image_list(image_folder)
        image_list = [os.path.join(image_folder,x[1]) for x in image_list]
        batch_size = 50
        img_embs = []

        for i in range(0,len(image_list),batch_size):
            image_batch = image_list[i:i+batch_size]
            images = []
            for x in image_batch:
                enhancer = ImageEnhance.Brightness(Image.open(x))
                brighter_image = enhancer.enhance(2.0)
                images.append(brighter_image)
            img_emb = model.encode(images)
            
            if serialize:
                for entry in img_emb:
                    img_embs.append(entry)
            else:
                img_embs.append(img_emb)

        print('image embedding generated')
        et = time.time()
        print("Time : ", et - st)

        return img_embs


    def get_image_list(self, image_folder):
        image_list = (os.listdir(image_folder))
        new_image_list = []
        hidden_ranges = set()
        data = self.read_json(os.path.join(os.path.dirname(image_folder),'data.json')).get('hidden_ranges',[])

        for l,r in data:
            for i in range(l,r+1):
                hidden_ranges.add(i)

        for x in image_list:
            _id_ = int(x.split('.')[0])
            if _id_ in hidden_ranges:
                continue
            new_image_list.append((_id_,image_folder + '/' + x))

        image_list = sorted(new_image_list)

        return image_list


    def get_captions(self, link):
        video_id = link.split("v=")[1]
        transcripts = YouTubeTranscriptApi.get_transcript(video_id)
        subtitle_text = " ".join([entry["text"] for entry in transcripts])
        return transcripts
    

    def create_transcript(self, yt_link, word_limit = 100):
        sentence_per_rank = 1
        cap = self.get_captions(yt_link)
        context = []
        current_context = {'text' : '', 'start' : 99999999, 'end' : -1}
        sen_count = 0
        for snap in cap:
            text = snap['text'].strip()
            current_context['text'] += ' ' + text
            # start = 
            current_context['start'] = min(current_context['start'], snap['start'])
            current_context['end'] = max(current_context['end'], snap['duration'] + snap['start'])
            
            if current_context['text'].endswith('.'):
                sen_count += 1
            
            if sen_count == sentence_per_rank:
                context.append(current_context)
                current_context = {'text' : '', 'start' : 99999999, 'end' : -1}
                sen_count = 0
        
        new_context = []
        for snap in context:
            text = snap['text'].strip()
            snap['word_count'] = len(text.split(' '))
            if len(new_context) == 0:
                new_context.append(snap)
            else:
                if new_context[-1]['word_count'] + snap['word_count'] > word_limit:
                    new_context.append(snap)
                else:
                    le = new_context.pop()
                    le['text'] = le['text'] + ' ' + snap['text']
                    le['end'] = snap['end']
                    le['word_count'] += snap['word_count']
                    new_context.append(le)
 
        return new_context


    def alter_transcript(self, transcript):
        altered_transcript = []
        for entry in transcript:
            text = entry['text']
            start = entry['start']
            end = entry['end']
            gpt_data = sm.generate_dict_from_chat_gpt(system,text,['data'])
            print(gpt_data['data'])
            altered_transcript.append({
                'text' : (gpt_data['data'], text),
                'start' : start,
                'end' : end
            })
        return altered_transcript


    def hidden_range_utility(self, text):
        text = text.split('\n')
        hidden_ranges_movie = []
        hidden_ranges_yt = []
        is_movie = True
        for r in text:
            if r == 'movie':
                continue
            elif r == 'yt':
                is_movie = False
            else:
                left, right = r.split(',')
                left = int(left)
                right = int(right)
                if is_movie:
                    hidden_ranges_movie.append((left,right))
                else:
                    hidden_ranges_yt.append((left,right))

        return hidden_ranges_movie, hidden_ranges_yt


    def test(self, youtube_link):
        transcript = self.create_transcript(youtube_link)
        if len(transcript) < 3:
            raise
        else:
            print("Test Passed!")


    def search(self, query, embs, top_k):
            best_match = (-10.0,-10)
            emb_count = 0
            for emb in embs:
                hit = util.semantic_search(query, emb, top_k = top_k)
                if hit[0][0]['score'] > best_match[0]:
                    best_match = (hit[0][0]['score'], emb_count + hit[0][0]['corpus_id'])
                emb_count += len(emb)
            return best_match[1]


    def synthesize(self, text = '', speaker = '', youtube_link = '', output = ''):
        
        self.test(youtube_link)

        hidden_ranges_movie, hidden_ranges_yt = self.hidden_range_utility(text)
        speaker = 'gpt_cove'
        _main_folder_path_ = self.get_main_folder_path()
        _folder_path_ = self.get_folder_path()
        image_folder_path = os.path.join(_main_folder_path_,'image')

        os.makedirs(_main_folder_path_, exist_ok = True)
        os.makedirs(image_folder_path, exist_ok = True)
   
        yt_path = os.path.join(_main_folder_path_, 'movie/yt_video.mp4')
        movie_path = os.path.join(_main_folder_path_, 'movie/movie.mp4')
        yt_images = os.path.join(image_folder_path, 'yt')
        movie_images = os.path.join(image_folder_path,'movie')

        print("Downloading youtube video")
        self.download_yt(youtube_link, yt_path)
        print("Downloading youtube video : DONE")

        print("Creating images")
        self.create_image(yt_path, yt_images, hidden_ranges_yt)
        self.create_image(movie_path, movie_images, hidden_ranges_movie)
        print("Creating images : DONE")



        print("Computing Embedding")
        query_emb = self.compute_embedding(yt_images, serialize = True)
        img_emb = self.compute_embedding(movie_images)
        print("Computing Embedding : DONE")

        n_image_list = self.get_image_list(os.path.join(movie_images, 'images'))
        top_k = 1
        timestamp = {}
        for i in range(len(query_emb)):
            best_match = self.search(query_emb[i], img_emb, top_k = top_k)
            print("SEARCH : {}".format(i))
            timestamp[i] = n_image_list[best_match][0]
        
        self.save_json(timestamp,os.path.join(_main_folder_path_,'timestamp.json'))

        transcript = self.create_transcript(youtube_link)        
        transcript = self.alter_transcript(transcript)
        print("Transcript created : ", len(transcript))
    
        audio_files = []

        for itr in range(len(transcript)):
            sen, original_speech = transcript[itr]['text']
            start = transcript[itr]['start']
            end = transcript[itr]['end']
            duration = end - start
            _hash_ = uuid.uuid4().hex
            audio_folder = os.path.join(_folder_path_, _hash_)
            movie_timestamps = self.predict_movie_timestamp(start, duration, timestamp)
            os.mkdir(audio_folder)
            audio_file = os.path.join(audio_folder, 'speech.wav')
            self.save_audio_instance(text = sen,
                                     file_path = audio_file,
                                     speaker_wav = speaker,
                                     silence = 0,
                                     rank = itr + 1,
                                     chapter = 1,
                                     start = start,
                                     duration = duration,
                                     movie_timestamps = movie_timestamps,
                                     original_movie_timestamps = movie_timestamps,
                                     original_speech = original_speech
                                    )


    def save_audio_instance(
                            self,
                            text : str = None,
                            file_path : str = None,
                            speaker_wav : str = None,
                            silence : float = None,
                            rank : int = None,
                            chapter : int = None,
                            start : float = None,
                            duration : float = None,
                            movie_timestamps = None,
                            original_movie_timestamps = None,
                            original_speech = None,
                            ):

            time = self.save_with_silence(text=text, file_path= file_path, speaker_wav=speaker_wav, silence = silence, duration = duration)
            asr_speech, wer, segment_timestamps = self.get_asr_and_wer(text, file_path)
            data = {
                'time' : time,
                'speaker' : speaker_wav,
                'rank' : rank,
                'chapter' : chapter,
                'speech' : text,
                'asr_speech' : asr_speech,
                'wer' : wer,
                'silence' : silence,
                'segment_timestamps' : segment_timestamps,
                'start' : start,
                'duration' : duration,
                'movie_timestamps' : movie_timestamps,
                'original_movie_timestamps' : original_movie_timestamps,
                'orginal_speech' : original_speech
            }
            data_path = file_path.replace("speech.wav","data.json")
            self.save_json(data, data_path)


    def predict_movie_timestamp(self, start, duration, timestamp):
        ts = []
        start = int(start)
        duration = int(duration)
        for time in range(start,start+duration,1):
            ts.append(timestamp[time])
        return ts


    def save_text(self, _ids_, text):
        if type(_ids_) != list:
            _ids_ = [_ids_]
        
        _folder_path_ = self.get_folder_path()

        _new_ids_ = []
        for _id_ in _ids_:
            _new_ids_.append(_id_.split('_')[-1])
        _ids_ = _new_ids_
        for _id_ in _ids_:
            
            audio_folder = os.path.join(_folder_path_, _id_)
            audio_data_path = os.path.join(audio_folder, 'data.json')
            audio_file = os.path.join(audio_folder, 'speech.wav')
            data = self.read_json(audio_data_path)
            data['speech'] = text
            self.save_audio_instance(text = data.get('speech'),
                            file_path = audio_file,
                            speaker_wav = data.get('speaker'),
                            silence = data.get('silence'),
                            rank = data.get('rank'),
                            chapter = data.get('chapter'),
                            start = data.get('start'),
                            duration = data.get('duration'),
                            movie_timestamps = data.get('movie_timestamps',None),
                            original_movie_timestamps = data.get('orginal_movie_timestamps',None),
                            original_speech = data.get('original_speech', None)
                            )
        return self.get_context(set(_ids_))


    def save_with_silence(self, text="", file_path = "", speaker_wav = "", silence = 250, duration = None):

        if text == "":
            current_audio = AudioSegment.silent(duration=silence)
        else:
            sm.extract_audio_from_chat_gpt(text, file_path)
            current_audio0 = AudioSegment.from_file(file_path)
            if silence > 0:
                current_audio = current_audio0 + AudioSegment.silent(duration=silence)
            else:
                current_audio = current_audio0
        time = len(current_audio)
        current_audio.export(file_path, format="wav") 
        return time


    def regenerate_by_ids(self, _ids_):
        if type(_ids_) != list:
            _ids_ = [_ids_]
        
        _folder_path_ = self.get_folder_path()

        _new_ids_ = []
        for _id_ in _ids_:
            _new_ids_.append(_id_.split('_')[-1])
        _ids_ = _new_ids_
        for _id_ in _ids_:
            
            audio_folder = os.path.join(_folder_path_, _id_)
            audio_data_path = os.path.join(audio_folder, 'data.json')
            audio_file = os.path.join(audio_folder, 'speech.wav')
            data = self.read_json(audio_data_path)
            self.save_audio_instance(text = data.get('speech'),
                            file_path = audio_file,
                            speaker_wav = data.get('speaker'),
                            silence = data.get('silence'),
                            rank = data.get('rank'),
                            chapter = data.get('chapter'),
                            start = data.get('start'),
                            duration = data.get('duration'),
                            movie_timestamps = data.get('movie_timestamps',None),
                            original_movie_timestamps = data.get('orginal_movie_timestamps',None),
                            original_speech = data.get('original_speech', None)
                            )
        return self.get_context(set(_ids_))

def test():
    q1 = QuotesGPTMiddleware("test")
    q1.synthesize("as")

    ctx = q1.get_context()
    ctx2 = q1.get_internal_context()
    import pdb; pdb.set_trace()


def test2():
    
    audio_wav = '/home/prakharrrr4/pegasus/spider/dan_clipped.wav'
    audio = whisper.load_audio(audio_wav)
    model = whisper.load_model("base")
    result = whisper.transcribe(model, audio)
    import pdb; pdb.set_trace()

def test3():
    q1 = QuotesGPTMiddleware(
        'test'
    )
    q1.synthesize('as')
    ret = q1.get_subtitles_at_word_level()
    import pdb; pdb.set_trace()


def test4():
    q1 = QuotesGPTMiddleware(
        'test'
    )
    q1.synthesize('as')
    _ids_ = ['t1','t1'] 
    q1.delete_by_ids(_ids_)


"""
Test Case 1
QuotesGPTMiddleware

1. synthesize -> working, filepath consistency ✅
2. object addition, save_file ✅
3. get_context, get_internal_context ✅
4. silence addition ✅
5. get subtitles in segment level ✅
__________________________________________

Test Case 2
Function Add-ons

0. regenerate_by_ids ✅
1. whisperx ✅
2. word level ts ✅
3. delete speech [I] ✅
4. get subtites at word level ✅

_____________________________________________

Test Case 3
UI Tests

1. delete function from UI ✅
2. movie and quotes UI flow 

_____________________________________________

Test Case 4
0. video file optimization ✅
1. preset_{i}.py impl. 

_____________________________________________

Test Case 5
Crawler(SM) Tests 
1. movie and quotes crawler ✅ 
"""
