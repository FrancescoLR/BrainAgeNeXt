#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Francesco La Rosa
"""
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, Spacingd, CropForegroundd, SpatialPadd, CenterSpatialCropd
from monai.data import CacheDataset
import numpy as np
import os
import torchio
import torch.nn as nn
import matplotlib.pyplot as plt
from nnunet_mednext import create_mednext_v1, create_mednext_encoder_v1


class MedNeXtEncReg(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MedNeXtEncReg, self).__init__()
        self.mednextv1 = create_mednext_encoder_v1(num_input_channels=1, num_classes=1, model_id='B', kernel_size=3, deep_supervision=True)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.regression_fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        mednext_out = self.mednextv1(x)
        x = mednext_out
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        age_estimate = self.regression_fc(x)
        return age_estimate.squeeze()


def prepare_transforms():
    x, y, z = (160, 192, 160)
    p = 1.0
    monai_transforms = [
        LoadImaged(keys=["image"], ensure_channel_first=True),
        Spacingd(keys=["image"], pixdim=(p, p, p)),
        CropForegroundd(keys=["image"], allow_smaller=True, source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(x, y, z)),
        CenterSpatialCropd(keys=["image"], roi_size=(x, y, z))
    ]
    val_torchio_transforms = torchio.transforms.Compose(
        [torchio.transforms.ZNormalization(masking_method=lambda x: x > 0, keys=["image"], include=['image'])]
    )
    return Compose(monai_transforms + [val_torchio_transforms])


def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df.dropna(subset=['Path'], inplace=True)
    df.dropna(subset=['Age'], inplace=True)
    data_dicts = [{'image': row['Path'], 'label': row['Age']} for index, row in df.iterrows()]
    return df, data_dicts


def create_dataloader(data_dicts, transforms):
    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=0.2, num_workers=4)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=torch.cuda.is_available())
    return dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model():
    torch.cuda.empty_cache()
    return MedNeXtEncReg().to(device)


def run_predictions(model_path, dataloader):
    model = initialize_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_data in dataloader:
            images = batch_data['image'].to(device)
            pred = model(images)
            predictions.append(pred.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.array(predictions)


def main(csv_file):
    df, data_dicts = load_data(csv_file)
    transforms = prepare_transforms()
    dataloader = create_dataloader(data_dicts, transforms)

    model_paths = [
        os.path.join(os.path.dirname(__file__), f'BrainAge_{i}.pth') for i in range(1, 6)
    ]

    predictions_list = [run_predictions(model_path, dataloader) for model_path in model_paths]
    average_predictions = np.median(np.stack(predictions_list), axis=0)

    CA = df['Age'].values
    BA = average_predictions.flatten()
    BA_corr = np.where(CA > 18, BA + (CA * 0.062) - 2.96, BA)
    BAD_corr = BA_corr - CA
    
    df['Predicted_Brain_Age'] = BA_corr
    df['Brain_Age_Difference'] = BAD_corr

    df.to_csv(csv_file.replace('.csv', '_with_predictions.csv'), index=False)
    print('Updated CSV file saved.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Error: No .csv file provided.")
        print("Usage: python script.py <path_to_csv_file>")
        print("Please provide the path to a .csv file as the argument. This file should contain columns for 'Path' and 'Age' for all subjects.")
        sys.exit(1)
    csv_file = sys.argv[1]
    main(csv_file)

