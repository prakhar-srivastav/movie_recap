from moviepy.editor import VideoFileClip, concatenate_videoclips
v1 = VideoFileClip('/home/prakharrrr4/wolfy_ui/wolf/media/workspace/danish_girl/movie/movie.mp4')
v1 = v1.resize(0.25)  
v1.write_videofile('/home/prakharrrr4/wolfy_ui/wolf/media/workspace/danish_girl/movie/movie_com.mp4', bitrate="800k")