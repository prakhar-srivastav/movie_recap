from moviepy.editor import VideoFileClip
import moviepy.video.fx.all  as vfx
import os
from PIL import Image, ImageFile

from tqdm import tqdm
import time
import psutil
from concurrent.futures import ProcessPoolExecutor
from transformers import CLIPProcessor, CLIPModel
import signal
import uuid
import shutil
import json
import subprocess


"""
I am using CLIP to identify obvious bad targets and remove them
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch


def kill_child_processes(parent_pid = os.getpid(), sig=signal.SIGTERM):
    import pdb; pdb.set_trace()
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)  # Get all child processes
    for process in children:
        print(f"Killing child process {process.pid}")
        process.send_signal(sig)


def _sentence_splitter_(
                    text,
                    ):
    import pysbd
    segmenter = pysbd.Segmenter(language="en", clean=False)
    sentences = segmenter.segment(text)
    return sentences





def divide_chunks(lst, n):
    chunk_size = len(lst) // n
    chunks = [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    remainder = len(lst) % n
    if remainder:
        for i in range(1, remainder + 1):
            chunks[-i].append(lst[-i])
    
    return chunks


def make_image_out_of_movies(video_path, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_file = os.path.join(output_dir,'data.json')
    image_folder = os.path.join(output_dir,'images')
    data = read_json(json_file)
    if data.get('video_path') == video_path:
        return 
    try:
        shutil.rmtree(image_folder)
    except: 
        pass
    data['video_path'] = video_path
    save_json(data,json_file)

    os.makedirs(image_folder, exist_ok = True)

    duration = int(VideoFileClip(video_path).duration)
    num_workers = 16
    inps = [i for i in range(0, duration, 1)]
    inps = divide_chunks(inps, num_workers)
    inps = [(video_path, x, image_folder) for x in inps]
    with ProcessPoolExecutor() as executor:
        list(executor.map(save_frame, inps))


def read_json(filename):
    try:
        j = open(filename, 'r')
        json_object =  json.loads(j.read())
        j.close()
        return json_object
    except:
        return dict()


def save_json(request_body, filename):
    json_object = json.dumps(request_body, indent=4)    
    with open(filename, "w") as outfile:
        outfile.write(json_object)


def generate_clip_scores(folder_path):

    folder_path = os.path.join(folder_path, 'images')
    embedding_path = os.path.join(folder_path, 'embedding')
    image_path_list = os.listdir(folder_path)
    image_path_list = [folder_path + '/' + x for x in image_path_list]

    try:
        shutil.rmtree(embedding_path)
    except:
        pass
    os.makedirs(embedding_path ,exist_ok = True)

    hash = uuid.uuid4().hex + '.json'
    save_json({
        'folder_path' : folder_path,
        'embedding_path' : embedding_path,
        'image_path_list' : image_path_list,
    },
    hash)

    arg = read_json(hash)

    folder_path = arg.get('folder_path')
    image_path_list = arg.get('image_path_list')


    CHUNK = 50
    for i in tqdm(range(0,len(image_path_list),CHUNK)):

        ma = 0
        command = ["python3", "generate_clip_score_worker.py", hash, str(i), str(i + CHUNK)]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            cur_data = read_json(result_path)
            ma += 1
        else:
            raise




def get_image_features(images):
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to("cuda")
    inputs = processor(images=images, return_tensors="pt")
    inputs = inputs.to("cuda")
    image_features = model.get_image_features(**inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.tolist()

    return image_features


def save_embedding(image_folder):
    image_list = os.listdir(image_folder)
    new_image_list = []
    for x in image_list:
        if x.endswith('json'):
            continue
        else:
            new_image_list.append(image_folder + '/' + x)
    image_list = new_image_list
    image_list = [Image.open(x) for x in image_list]
    sz = 50
    for i in range(0,len(image_list),sz):
        import pdb; pdb.set_trace()
        ft = get_image_features(image_list[i:i+sz])    
        print(i)
        import pdb; pdb.set_trace()


def find_similartiy(n_image, q_image):
    n_image_list = os.listdir(n_image)
    q_image_list = os.listdir(q_image)

    new_n_image_list = list()
    for x in n_image_list:
        if x.endswith('json'):
            continue
        else:
            new_n_image_list.append(n_image + '/' + x)
    n_image_list = new_n_image_list
    n_image_list = [Image.open(x) for x in n_image_list]

    new_q_image_list = list()
    for x in q_image_list:
        if x.endswith('json'):
            continue
        else:
            new_q_image_list.append(q_image + '/' + x)
    q_image_list = new_q_image_list
    q_image_list = [Image.open(x) for x in q_image_list]

    # n_image_feature = get_image_features(n_image_list)
    q_image_feature = get_image_features(q_image_list)

    import pdb; pdb.set_trace()
    print(len(images))


from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import torch
import pickle
import zipfile
import os
from tqdm.autonotebook import tqdm
torch.set_num_threads(4)


def search(query, img_emb, k=1):
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(query_emb, img_emb, top_k=k)[0]
    import pdb; pdb.set_trace()    
    for hit in hits:
        print(img_names[hit['corpus_id']])
        display(IPImage(os.path.join(img_folder, img_names[hit['corpus_id']]), width=200))


def compute_embedding(image_folder):
    st = time.time()
    print('starting to generate image embedding')
    model = SentenceTransformer('clip-ViT-B-32')
    image_list = get_image_list(image_folder)
    images = [Image.open(x).resize((224,224)) for x in image_list]
    img_emb = model.encode(images)
    print('image embedding generated')
    et = time.time()
    print("Time : ", et - st)
    return img_emb


def get_image_list(image_folder):
    image_list = (os.listdir(image_folder))
    new_image_list = []
    for x in image_list:
        if x.endswith('json'):
            continue
        else:
            new_image_list.append((int(x.split('.')[0]),image_folder + '/' + x))

    image_list = sorted(new_image_list)
    image_list = [x[1] for x in image_list]
    return image_list


if __name__ == '__main__':

    # video_path = '/home/prakharrrr4/wolfy_ui/wolf/media/workspace/danish_girl/movie/movie_new_mutant.mp4'
    # folder_path = '/home/prakharrrr4/wolfy_ui/wolf/workspace/video_maker/image_of_movies_1'
    # yt_movie = '/home/prakharrrr4/wolfy_ui/wolf/workspace/video_maker/yt_movie.mp4'
    folder_path_a = '/home/prakharrrr4/wolfy_ui/wolf/workspace/video_maker/image_of_movies/'
    folder_path_b = '/home/prakharrrr4/wolfy_ui/wolf/workspace/video_maker/image_of_movies_1/images'
img_emb = compute_embedding(folder_path_a)
query_emb = compute_embedding(folder_path_b)
query_image_list = get_image_list(folder_path_b)
top_k = 1
timestamp = {}
for i in range(len(query_image_list)):
    x = query_emb[i]
    hits = util.semantic_search(x, img_emb, top_k=top_k)[0]
    timestamp[i] = hits[0]['corpus_id']

    import pdb; pdb.set_trace()
    # st = time.time()
    # generate_clip_scores(folder_path_b)
    # et = time.time()


    print("TIME : ", et - st)
    import pdb; pdb.set_trace()


