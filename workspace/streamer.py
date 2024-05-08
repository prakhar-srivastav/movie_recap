import streamlit as st

def main():
    st.title("Local Video Server")
    video_file_path = "/home/prakharrrr4/wolfy_ui/wolf/media/workspace/danish_girl/movie/movie.mp4"
    st.video(video_file_path)


if __name__ == "__main__":
    main()

