
# get filename from argument
filename=$1
if [ -z "$filename" ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi
asciinema rec terminal_recordings/cast/$filename.cast --append
