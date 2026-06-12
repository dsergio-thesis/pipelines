
filename="$1"
ffmpeg -i terminal_recordings/$filename.mkv \
       -filter:v "setpts=0.25*PTS" \
       -an \
       terminal_recordings/$filename-timelapse.mp4
