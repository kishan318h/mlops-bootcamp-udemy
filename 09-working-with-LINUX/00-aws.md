## Setup AWS CLI
- After installing AWS CLI, check AWS version using `aws --version`
- Create an access key for the AWS user
- In CLI, type `aws configure`:
    - copy and paste the AWS access key id and AWS secret access key
    - enter the default region (Eg: us-east-1)

## Create S3 bucket using CLI
- the bucket name should be globally unique
- `aws s3api create-bucket  --bucket <bucket name>`
- to copy files from local to s3 bucket:
    - `aws s3 cp <filename> s3://<bucket name>`
- to view the contents of a bucket:
    - `aws s3api list-objects-v2 --bucket <bucket name>`
- to delete the bucket: 
    - delete the files `aws s3 rm s3://<bucket name>/filename`
    - delete the bucket after the bucket is empty `aws s3api delete-bucket --bucket <bucket name>`
- Versioning in s3:
    - by default versioning is disabled in s3 bucket
    - if enabled, when we push a file with same name in the bucket, s3 will create a new version for it
    - if we need to restore an old version, we need to delete the newer version of file from console

