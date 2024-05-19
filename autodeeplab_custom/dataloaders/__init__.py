from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, sealer
from torch.utils.data import DataLoader
from torchvision import transforms

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set1, train_set2 = cityscapes.sp(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set1.NUM_CLASSES
        train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=True, **kwargs)
        train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        #test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        #return train_loader1, train_loader2, val_loader, test_loader, num_class
        return train_loader1, train_loader2, val_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, train_loader, val_loader, test_loader, num_class
    
    elif args.dataset == 'sealer':
        transform = transforms.Compose([
        # transforms.Resize(args.input_size),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        ])

        datasets = sealer.Sealer(args, '/home/ml/HeonSung/segmentation-nas/data/image', transform=transform)
        num_class = datasets.NUM_CLASSES
        train_set, val_set, test_set = sealer.split_dataset(datasets)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

