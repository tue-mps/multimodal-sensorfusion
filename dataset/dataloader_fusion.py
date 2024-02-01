import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
import torch

Sequences = {'Validation': ['RECORD@2020-11-22_12.49.56', 'RECORD@2020-11-22_12.11.49', 'RECORD@2020-11-22_12.28.47',
                            'RECORD@2020-11-21_14.25.06'],
             'Test': ['RECORD@2020-11-22_12.45.05', 'RECORD@2020-11-22_12.25.47', 'RECORD@2020-11-22_12.03.47',
                      'RECORD@2020-11-22_12.54.38']}

def RADIal_collate_standalone(batch):
    datas = []
    segmaps = []
    out_labels = []
    box_labelss = []
    img_names_ = []

    for (data, segmap, out_label, box_labels, img_name_) in batch:
        datas.append(torch.tensor(data).permute(2, 0, 1))
        segmaps.append(torch.tensor(segmap))
        out_labels.append(torch.tensor(out_label))
        box_labelss.append(torch.from_numpy(box_labels))
        img_names_.append(img_name_)

    return (torch.stack(datas), torch.stack(segmaps),
            torch.stack(out_labels), box_labelss, img_names_
            )

def RADIal_collate_earlyfusion_P(batch):
    fused_datas = []
    segmaps = []
    out_labels = []
    img_names_ = []

    for (fused_data, segmap, out_label, img_name_) in batch:
        fused_datas.append(torch.tensor(fused_data).permute(2, 0, 1))
        segmaps.append(torch.tensor(segmap)) #.permute(2, 0, 1))
        out_labels.append(torch.tensor(out_label))
        img_names_.append(img_name_)

    return (torch.stack(fused_datas), torch.stack(segmaps),
            torch.stack(out_labels), img_names_
            )

def RADIal_collate_Non_earlyfusion_P(batch):
    voxel_point_clouds = []
    camera_images = []
    segmaps = []
    out_labels = []
    img_names_ = []

    for (voxel_point_cloud, camera_image, segmap, out_label, img_name_) in batch:
        voxel_point_clouds.append(torch.tensor(voxel_point_cloud).permute(2, 0, 1))
        camera_images.append(torch.tensor(camera_image).permute(2,0,1))
        segmaps.append(torch.tensor(segmap)) #.permute(2, 0, 1))
        out_labels.append(torch.tensor(out_label))
        img_names_.append(img_name_)

    return (torch.stack(voxel_point_clouds),
            torch.stack(camera_images), torch.stack(segmaps),
            torch.stack(out_labels), img_names_
            )

def RADIal_collate_allfusion_B(batch):
    radar_FFTs = []
    bev_images = []
    segmap_polars = []
    out_labels = []
    box_labelss = []
    bev_img_names = []

    for (radar_FFT, bev_image, segmap_polar, out_label, box_labels, bev_img_name) in batch:
        radar_FFTs.append(torch.tensor(radar_FFT).permute(2, 0, 1))
        bev_images.append(torch.tensor(bev_image).permute(2, 0, 1))
        segmap_polars.append(torch.tensor(segmap_polar)) #.permute(2, 0, 1))
        out_labels.append(torch.tensor(out_label))
        box_labelss.append(torch.from_numpy(box_labels))
        bev_img_names.append(bev_img_name)

    return (torch.stack(radar_FFTs), torch.stack(bev_images),
            torch.stack(segmap_polars), torch.stack(out_labels),
            box_labelss, bev_img_names
            )

def get_collate_function(config):
    # B stands for BEV
    # P stands for Perspective
    if config['architecture']['perspective']['early_fusion'] == 'True':
        return RADIal_collate_earlyfusion_P

    if (config['architecture']['perspective']['after_decoder_fusion'] == 'True' or
        config['architecture']['perspective']['x4_fusion'] == 'True'):
        return RADIal_collate_Non_earlyfusion_P

    if (config['architecture']['perspective']['only_camera'] == 'True' or
        config['architecture']['bev']['only_radar'] == 'True'):
        return RADIal_collate_standalone

    if (config['architecture']['bev']['only_radar'] == 'False'  or
        config['architecture']['bev']['after_decoder_fusion'] == 'True' or
        config['architecture']['bev']['x4_fusion'] == 'True'):
        return RADIal_collate_allfusion_B

def CreateDataLoaders(dataset, config=None, seed=0):
    if (config['dataloader']['mode'] == 'random'):
        # generated training and validation set
        # number of images used for training and validation
        n_images = dataset.__len__()

        split = np.array(config['dataloader']['split'])
        if (np.sum(split) != 1):
            raise NameError('The sum of the train/val/test split should be equal to 1')
            return

        n_train = int(config['dataloader']['split'][0] * n_images)
        n_val = int(config['dataloader']['split'][1] * n_images)
        n_test = n_images - n_train - n_val

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))

        print('===========  Dataset  ==================:')
        print('      Mode:', config['dataloader']['mode'])
        print('      Train Val ratio:', config['dataloader']['split'])
        print('      Training:', len(train_dataset), ' indexes...', train_dataset.indices[:3])
        print('      Validation:', len(val_dataset), ' indexes...', val_dataset.indices[:3])
        print('      Test:', len(test_dataset), ' indexes...', test_dataset.indices[:3])
        print('')

        # create data_loaders
        train_loader = DataLoader(train_dataset,
                                  batch_size=config['dataloader']['train']['batch_size'],
                                  shuffle=True,
                                  num_workers=config['dataloader']['train']['num_workers'],
                                  pin_memory=True,
                                  collate_fn=get_collate_function(config))
        val_loader = DataLoader(val_dataset,
                                batch_size=config['dataloader']['val']['batch_size'],
                                shuffle=False,
                                num_workers=config['dataloader']['val']['num_workers'],
                                pin_memory=True,
                                collate_fn=get_collate_function(config))
        test_loader = DataLoader(test_dataset,
                                 batch_size=config['dataloader']['test']['batch_size'],
                                 shuffle=False,
                                 num_workers=config['dataloader']['test']['num_workers'],
                                 pin_memory=True,
                                 collate_fn=get_collate_function(config))

        return train_loader, val_loader, test_loader

    elif (config['dataloader']['mode'] == 'sequence'):
        dict_index_to_keys = {s: i for i, s in enumerate(dataset.sample_keys)}

        Val_indexes = []
        for seq in Sequences['Validation']:
            idx = np.where(dataset.labels[:, 14] == seq)[0]
            Val_indexes.append(dataset.labels[idx, 0])
        Val_indexes = np.unique(np.concatenate(Val_indexes))

        Test_indexes = []
        for seq in Sequences['Test']:
            idx = np.where(dataset.labels[:, 14] == seq)[0]
            Test_indexes.append(dataset.labels[idx, 0])
        Test_indexes = np.unique(np.concatenate(Test_indexes))

        val_ids = [dict_index_to_keys[k] for k in Val_indexes]
        test_ids = [dict_index_to_keys[k] for k in Test_indexes]
        train_ids = np.setdiff1d(np.arange(len(dataset)), np.concatenate([val_ids, test_ids]))

        train_dataset = Subset(dataset, train_ids)
        val_dataset = Subset(dataset, val_ids)
        test_dataset = Subset(dataset, test_ids)

        print('===========  Dataset  ==================:')
        print('      Mode:', config['dataloader']['mode'])
        print('      Training:', len(train_dataset))
        print('      Validation:', len(val_dataset))
        print('      Test:', len(test_dataset))
        print('')

        # create data_loaders
        train_loader = DataLoader(train_dataset,
                                  batch_size=config['dataloader']['train']['batch_size'],
                                  shuffle=True,
                                  num_workers=config['dataloader']['train']['num_workers'],
                                  pin_memory=True,
                                  collate_fn=get_collate_function(config))
        val_loader = DataLoader(val_dataset,
                                batch_size=config['dataloader']['val']['batch_size'],
                                shuffle=False,
                                num_workers=config['dataloader']['val']['num_workers'],
                                pin_memory=True,
                                collate_fn=get_collate_function(config))
        test_loader = DataLoader(test_dataset,
                                 batch_size=config['dataloader']['test']['batch_size'],
                                 shuffle=False,
                                 num_workers=config['dataloader']['test']['num_workers'],
                                 pin_memory=True,
                                 collate_fn=get_collate_function(config))

        return train_loader, val_loader, test_loader

    else:
        raise NameError(config['dataloader']['mode'], 'is not supported !')
        return
