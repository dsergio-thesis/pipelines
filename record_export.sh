
filename=$1
if [ -z "$filename" ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi
agg --speed 4 terminal_recordings/cast/$filename.cast terminal_recordings/$filename.gif
