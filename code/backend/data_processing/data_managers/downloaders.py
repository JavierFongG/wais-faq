import boto3
import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

class Downloader(ABC):
    @abstractmethod
    def download(self, file_path) -> bytes:
        pass


class S3Downloader(Downloader):

    def __init__(self, bucket_name : str):
        self.s3 = boto3.client(
            's3'
            , aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            , aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        self.bucket_name = bucket_name

    def download(self, file_path : str) -> bytes:
        response = self.s3.get_object(
            Bucket = self.bucket_name
            , Key = file_path
        )
        return response['Body'].read()
