import albumentations as A
from albumentations.pytorch import ToTensorV2


def transform():
    transform = A.Compose(
        [   
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ]
    )
    return transform