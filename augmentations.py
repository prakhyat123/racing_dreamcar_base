import torchvision.transforms as transforms

from lanemarkingfilter import color_filter

class LaneMarking(object):
    def __call__(self, img):
        return color_filter(img)

class TrainTransform(object):
    def __init__(self):
        self.transform_prime = transforms.Compose(
            [   
                transforms.Resize(256),
                transforms.CenterCrop(224),
                LaneMarking(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ]
            )

        self.transform= transforms.Compose(
            [   transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ]
            )
        pass

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2



