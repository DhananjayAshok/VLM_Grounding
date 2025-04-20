storage_dir=$1
if [ -z "$storage_dir" ]; then
  echo "Usage: $0 <storage_dir>"
  exit 1
fi
# Download the landmarks
cd $storage_dir
# check that data/raw exists, if not error out
if [ ! -d "data/raw" ]; then
  echo "data/raw directory does not exist. Please create it and try again."
  exit 1
fi
cd data/raw

exit 1
gdown  --fuzzy  # link is not added yet. Will add later because I fear it might not be anonymized. 
unzip landmark_images.zip