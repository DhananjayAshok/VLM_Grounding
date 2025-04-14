storage_dir=$1

if [ -z "$storage_dir" ]; then
  echo "Usage: $0 <storage_dir>"
  exit 1
fi

# if raw subdir does not exist throw error
if [ ! -d "$storage_dir/data/raw" ]; then
  echo "$storage_dir/data/raw directory does not exist. Please create it first."
  exit 1
fi

mkdir -p $storage_dir/data/raw/okvqa
cd $storage_dir/data/raw/okvqa
wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip
unzip mscoco_val2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip
unzip OpenEnded_mscoco_val2014_questions.json.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm *.zip

python main.py setup_okvqa