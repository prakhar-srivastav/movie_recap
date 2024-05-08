
from moviepy.editor import AudioClip, VideoClip, AudioFileClip, VideoFileClip, CompositeAudioClip, CompositeVideoClip, ImageSequenceClip, ImageClip
from moviepy.editor import concatenate_audioclips, concatenate_videoclips
import moviepy.video.fx.all as vfx 

video1 = VideoFileClip("/home/prakharrrr4/wolfy_ui/wolf/workspace/video_maker/temp/movies_snips/student.mp4.mp4")
video2 = video1.subclip(10,20)
video3 = VideoFileClip("/home/prakharrrr4/wolfy_ui/wolf/workspace/video_maker/assets/video_affects/cloud.mp4")
video3 = video3.set_duration(video2.duration)
video3 = video3.resize(video2.size)
video3 = video3.set_opacity(0.1)
video4 = CompositeVideoClip([video2,video3])
video4.write_videofile("vid.mp4", codec="libx264", ffmpeg_params=["-preset", "fast"])



"""
I N V E R T
"""
