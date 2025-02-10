import os
import time
import uuid
import logging

import boto3
import botocore

logger = logging.getLogger(__name__)

def initialize_s3_client(xw
        profile_name,
        region_name=None,
        return_session=False):
    """
    Initialize boto3 session and S3 client.
    
    Args:
        profile_name (str): AWS profile name.
        return_session (bool): Return boto3 session if True.
    Returns:
        boto3.client: S3 boto3 client.
    """
    try:
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        logger.debug("Created boto3 session")
    except Exception as e:
        logger.error(f"Failed to create boto3 session: {e}")
        return
    
    try:
        boto_config = botocore.config.Config(max_pool_connections=50)
        s3_client = session.client("s3", config=boto_config, use_ssl=False)
        logger.debug("Created S3 client")
        logger.debug(f"Available buckets: {s3_client.list_buckets().get('Buckets')}")
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        return
    if return_session:
        return s3_client, session
    return s3_client


def wipe_bucket_dir(s3_client, bucket_name, bucket_prefix=""):
    """
    Deletes all files under a specific prefix in an S3 bucket.

    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): Name of the S3 bucket.
        bucket_prefix (str): Prefix (directory) to delete.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    files_deleted = 0
    try:
        pages = paginator.paginate(Bucket=bucket_name, Prefix=bucket_prefix)
        for page in pages:
            if "Contents" in page:
                try:
                    objects = [{"Key": obj["Key"]} for obj in page["Contents"]]
                    s3_client.delete_objects(Bucket=bucket_name, Delete={"Objects": objects})
                    files_deleted += len(objects)
                    logger.debug(f"Deleted {len(objects)} files")
                except Exception as e:
                    logger.error(f"Failed to delete files in page {page}: {e}")
                    return False
        logger.debug(f"Deleted {files_deleted} files under s3://{bucket_name}/{bucket_prefix}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete files under s3://{bucket_name}/{bucket_prefix}: {e}")
        return False


def wipe_bucket(s3_client, bucket_name):
    """
    Clear all files in the S3 bucket.

    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): Name of the S3 bucket.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    files = list_bucket_files(s3_client, bucket_name)
    if files is None:
        logger.warning(f"Skipping bucket cleanup: Failed to list files in s3://{bucket_name}/")
        return False
    elif len(files) > 0:
        logger.info(f"Found {len(files)} files in s3://{bucket_name}/")
        success = wipe_bucket_dir(s3_client, bucket_name)
        if not success:
            logger.error(f"Cannot proceed. Files not deleted from s3://{bucket_name}/")
            return False
    return True


def list_bucket_files(s3_client, bucket_name, bucket_prefix=""):
    """
    Get a list of all files in an S3 bucket under a given prefix.

    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): Name of the S3 bucket.
        bucket_prefix (str): S3 prefix (folder) to list objects from.

    Returns:
        dict: {file_name: file_size_in_bytes} for all files in S3.
    """            
    existing_files = {}
    paginator = s3_client.get_paginator("list_objects_v2")

    try:
        pages = paginator.paginate(Bucket=bucket_name, Prefix=bucket_prefix)
        for page in pages:
            if "Contents" in page:
                for obj in page.get("Contents", []):
                    existing_files[obj["Key"]] = obj["Size"]
        if not existing_files:
            logger.debug(f"No files found in s3://{bucket_name}/{bucket_prefix}")
    except Exception as e:
        logger.error(f"Failed to list files in s3://{bucket_name}/{bucket_prefix}: {e}")
    return existing_files


def generate_s3_key(file_path, separator="::"):
    """
    Generate a unique s3 key using a short UUID, the preceeding directory name,
    and the base name of the file.
    
    Args:
        file_path (str): path to the file.
        separator (str): separator between the directory name and the base name of the file.
        
    Returns:
        str: unique S3 key.
    """
    base_name = os.path.basename(file_path)
    dir_name = os.path.basename(os.path.dirname(file_path))
    short_uuid = uuid.uuid4().hex[:6]

    key = f"{short_uuid}{separator}{dir_name}{separator}{base_name}"
    return key


def upload_large_files_to_bucket(
        s3_client,
        bucket_name, 
        files_list,
        file_names=None,
        prefix="raw",
        ext="",
        reupload=False,
        threshold=20 * 1024 * 1024,
        chunk_size=20 * 1024 * 1024,
        max_concurrency=5):
    """
    Upload large files to S3 bucket using multipart upload.
        
    Args:
        s3_client (boto3.client): S3 boto3 client.
        bucket_name (str): Name of the S3 bucket.
        files_list (list): List of file paths to upload.
        file_names (list): List of file names to use in S3.
        prefix (str): S3 key prefix.
        ext (str): File extension to filter files.
        reupload (bool): Reupload files even if they exist.
        threshold (int): Multipart upload threshold in bytes.
        chunk_size (int): Multipart upload chunk size in bytes.
        max_concurrency (int): Maximum number of concurrent uploads.
    """

    config = boto3.s3.transfer.TransferConfig(
        multipart_threshold=threshold,
        multipart_chunksize=chunk_size,
        max_concurrency=max_concurrency,
        use_threads=True,
    )
    logger.debug(f"Using multipart_threshold: {(threshold)/(1024*1024):.2f} MB, chunk size: {(chunk_size)/(1024 * 1024):.2f} MB, max concurrency: {max_concurrency}")
    
    existing_files = list_bucket_files(s3_client, bucket_name, prefix)
    start_time = time.time()
    count = 0
    total_files = len(files_list)
    if file_names is None or len(file_names) != len(files_list):
        file_names = [generate_s3_key(file_path) for file_path in files_list]
        logger.warning("file_names not provided or length mismatch. Using base names of files_list.")

    for file_path, file_name in zip(files_list, file_names):
        file_exists = os.path.exists(file_path) and os.path.isfile(file_path)

        if file_exists and (not ext or file_path.endswith(ext)):
            s3_key = f"{prefix}/{file_name}"
            local_size = os.path.getsize(file_path)

            if not reupload and s3_key in existing_files and existing_files[s3_key] == local_size:                
                count += 1
                total_time_str = time.strftime("%M:%S", time.gmtime((time.time() - start_time)))
                logger.debug(f"({total_time_str}) ({count}/{total_files}) Skipping: {file_name}")
                continue
            
            try:
                count += 1
                s3_client.upload_file(file_path, bucket_name, f"{prefix}/{file_name}", Config=config)
                total_time_str = time.strftime("%M:%S", time.gmtime((time.time() - start_time)))
                logger.debug(f"({total_time_str}) ({count}/{total_files}) Uploaded: {file_name} to s3://{bucket_name}/{prefix}/")
            except Exception as e:
                logger.error(f"Failed to upload {file_name}: {e}")
        else:
            logger.warning(f"Skipping: {file_path} (File not found or invalid)")
