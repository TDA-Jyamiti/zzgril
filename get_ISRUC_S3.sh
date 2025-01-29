mkdir -p ./rawdata/ISRUC_S3/ExtractedChannels
mkdir -p ./rawdata/ISRUC_S3/RawData
echo 'Make data dir: ./rawdata/ISRUC_S3'

cd ./rawdata/ISRUC_S3/RawData
for s in $(seq 1 10):
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupIII/$s.rar
    unrar x $s.rar
done
echo 'Download Data to "./rawdata/ISRUC_S3/RawData" complete.'

cd ./rawdata/ISRUC_S3/ExtractedChannels
for s in $(seq 1 10):
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupIII-Extractedchannels/subject$s.mat
done
echo 'Download ExtractedChannels to "./rawdata/ISRUC_S3/ExtractedChannels" complete.'
