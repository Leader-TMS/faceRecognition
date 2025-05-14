import boto3
from botocore.client import Config
import os
import logging
#   STAGING
minio_endpoint = 'https://s3-staging-api.datacenter.tms-s.vn'  
access_key = 'xXDO5QNLCkhahlNE0Rm7'
secret_key = '2cvES9wanRg5B7J4SK03AiEHcGr4lrbM5rR0BSc2'
bucket_name = 'computer-vision'
folder_prefix = 'dataset/'
local_directory = 'dataset/'

#   PROD
# minio_endpoint = 'https://s3-api.app.ftcjsc.com'  
# access_key = 'YHYoxTd6ZdzLRumj'
# secret_key = 'bkC1i3FYyNTNL7VYQcg0imtxJRD97nfW'
# bucket_name = 'ttd'
# folder_prefix = 'gallery/'
# local_directory = 'minio/'


logging.basicConfig(level=logging.INFO)
imageExtensions = {'.jpg', '.png'}
s3Client = boto3.client(
    's3',
    endpoint_url=minio_endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    config=Config(signature_version='s3v4')
)

def getLocalFilesInfo(directory):
    filesInfo = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            filePath = os.path.join(root, file)
            mtime = os.path.getmtime(filePath)
            relativePath = os.path.relpath(filePath, directory)
            filesInfo[relativePath] = mtime
    return filesInfo

def getS3FilesInfo(bucket, prefix=''):
    filesInfo = {}
    continuationToken = None
    while True:
        kwargs = {'Bucket': bucket, 'Prefix': prefix}
        if continuationToken:
            kwargs['ContinuationToken'] = continuationToken

        response = s3Client.list_objects_v2(**kwargs)
        if 'Contents' in response:
            for obj in response['Contents']:
                if any(obj['Key'].lower().endswith(ext) for ext in imageExtensions):
                    filesInfo[obj['Key']] = obj['LastModified'].timestamp()

        continuationToken = response.get('NextContinuationToken')
        if not continuationToken:
            break
    return filesInfo

def downloadFiles(bucket = bucket_name, localDirectory = local_directory):
    localFilesInfo = getLocalFilesInfo(localDirectory)
    s3FilesInfo = getS3FilesInfo(bucket, 'dataset/')

    filesToDownload = [
        fileKey for fileKey, s3Mtime in s3FilesInfo.items()
        if os.path.relpath(fileKey, localDirectory) not in localFilesInfo or localFilesInfo[os.path.relpath(fileKey, localDirectory)] < s3Mtime
    ]

    total = len(filesToDownload)
    item  = 1
    for fileKey in filesToDownload:
        localPath = os.path.join(fileKey)
        os.makedirs(os.path.dirname(localPath), exist_ok=True)  
        
        try:
            s3Client.download_file(bucket, fileKey, localPath)
            logging.info(f'{item}/{total} - Downloaded: {fileKey}')
            item += 1
        except Exception as e:
            logging.error(f'Error downloading {fileKey}: {e}')

def uploadFiles(bucket=bucket_name, localDirectory=local_directory):
    localFilesInfo = getLocalFilesInfo(localDirectory)
    s3FilesInfo = getS3FilesInfo(bucket, folder_prefix)
    filesToUpload = [
        localFile for localFile, localMtime in localFilesInfo.items()
        if localDirectory + localFile not in s3FilesInfo or s3FilesInfo[localDirectory + localFile] < localMtime
    ]

    total = len(filesToUpload)
    item = 1
    for localFile in filesToUpload:
        localPath = os.path.join(localDirectory, localFile)
        s3Key = f'{folder_prefix}{localFile}'

        # Lấy thư mục chứa file local
        localDir = os.path.dirname(localPath)
        if not os.path.exists(localDir):
            os.makedirs(localDir)  # Tạo thư mục con nếu chưa có

        try:
            # Upload file lên S3
            s3Client.upload_file(localPath, bucket, s3Key)
            logging.info(f'{item}/{total} - Uploaded: {localFile}')
            item += 1
        except Exception as e:
            logging.error(f'Error uploading {localFile}: {e}')

if __name__ == "__main__":
    # downloadFiles()
    uploadFiles()