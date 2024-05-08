from moviepy.editor import VideoFileClip
import moviepy.video.fx.all  as vfx
import os
from PIL import Image, ImageFile
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time
import psutil
from transformers import CLIPProcessor, CLIPModel, TFCLIPVisionModel
import signal
import sys
import json

def read_json(filename):
    j = open(filename, 'r')
    json_object =  json.loads(j.read())
    j.close()
    return json_object


def save_json(request_body, filename):
    json_object = json.dumps(request_body, indent=4)    
    with open(filename, "w") as outfile:
        outfile.write(json_object)

def _sentence_splitter_(
                    text,
                    ):
    import pysbd
    segmenter = pysbd.Segmenter(language="en", clean=False)
    sentences = segmenter.segment(text)
    return sentences


def generate_clip_scores(hash, st, et):

    arg = read_json(hash)

    folder_path = arg.get('folder_path')
    image_path_list = arg.get('image_path_list')
    embedding_path = arg.get('embedding_path')
    image_path_list = os.listdir(folder_path)[st:en]

    image0 = [folder_path + '/' + x for x in image_path_list]
    images = [Image.open(x) for x in image0]

    print(len(images))

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to("cuda")
    import pdb; pdb.set_trace()
    inputs = processor(images=images, return_tensors="tf")
    image_features = model.get_image_features(**inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.tolist()
    import pdb; pdb.set_trace()
    return image_features








if __name__ == '__main__':
    # hash = sys.argv[1]
    # st = int(sys.argv[2])
    # en = int(sys.argv[3])
    hash = '630917d8b67444e69d8411a4e2537420.json'
    test_image_search(hash)
    # st = 1
    # en = 4

    # generate_clip_scores(hash, st, en)