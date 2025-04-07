storage_dir=$1

if [ -z "$storage_dir" ]; then
  echo "Usage: $0 <storage_dir>"
  exit 1
fi

cd $storage_dir
