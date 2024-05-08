import uuid
import os
import threading
import time
from TTS.api import TTS
import uuid


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
        import pysbd
        segmenter = pysbd.Segmenter(language="en", clean=False)
        sentences = segmenter.segment(text)
        return sentences


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

