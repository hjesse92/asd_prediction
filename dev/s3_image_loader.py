import boto3
import torch
from torchvision import transforms, datasets
from PIL import Image
from io import BytesIO

class S3ImageFolder(torch.utils.data.Dataset):
    def __init__(self, s3_bucket, s3_prefix, transform=None, augmented=True):
        self.s3 = boto3.resource('s3')
        self.bucket = self.s3.Bucket(s3_bucket)
        self.s3_prefix = s3_prefix
        self.transform = transform
        
        self.image_files = [obj.key for obj in self.bucket.objects.filter(Prefix=self.s3_prefix)]
        
        if not augmented:
            self.image_files = [f for f in self.image_files if not f.split('/')[-1].startswith("img")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        s3_object = self.bucket.Object(self.image_files[idx])
        img = Image.open(BytesIO(s3_object.get()['Body'].read()))
        
        if self.transform:
            img = self.transform(img)

        return img

def load_data(augment=False):
    s3_bucket = "w210-fall-2023-asd"
    train_asd_prefix = "asd_images/train/asd/"
    train_nonasd_prefix = "asd_images/train/nonasd/"
    
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            # any other transformations you'd like for augmentation
        ])
    else:
        transform_train = transforms.ToTensor()

    train_asd_data = S3ImageFolder(s3_bucket=s3_bucket, s3_prefix=train_asd_prefix, transform=transform_train, augmented=augment)
    train_nonasd_data = S3ImageFolder(s3_bucket=s3_bucket, s3_prefix=train_nonasd_prefix, transform=transform_train, augmented=augment)

    return train_asd_data, train_nonasd_data

# Usage
train_asd_data, train_nonasd_data = load_data(augment=True)

print(train_asd_data)
