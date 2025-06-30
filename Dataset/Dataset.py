class EEGDataset(Dataset):
    def __init__(self, task, split, base_path='./kaggle/input/mtcaic3', apply_noise=True, apply_trial_norm=True, apply_global_std=False, global_mean=None, global_std=None):
        self.task = task
        self.split = split
        self.base_path = base_path
        self.apply_noise = apply_noise
        self.apply_trial_norm = apply_trial_norm
        self.apply_global_std = apply_global_std
        self.global_mean = global_mean
        self.global_std = global_std

        self.meta_df = pd.read_csv(os.path.join(base_path, f'{split}.csv'))
        self.meta_df = self.meta_df[self.meta_df['task'] == task]

        self.label_encoder = LabelEncoder()
        if split != 'test':
            self.meta_df['label_enc'] = self.label_encoder.fit_transform(self.meta_df['label'])

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        eeg_path = os.path.join(
            self.base_path,
            row['task'],
            self.split,
            row['subject_id'],
            str(row['trial_session']),
            'EEGdata.csv'
        )

        df = pd.read_csv(eeg_path)

        samples_per_trial = 2250 if row['task'] == 'MI' else 1750
        trial_num = int(row['trial'])
        start_idx = (trial_num - 1) * samples_per_trial
        end_idx = start_idx + samples_per_trial

        df = df.iloc[start_idx:end_idx]
        eeg = df[['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']].values
        #eeg = df[['C3', 'CZ', 'C4']].values

        eeg = bandpass_filter(eeg)

        # Apply per-trial z-score normalization
        if self.apply_trial_norm:
            eeg = normalize_per_trial(eeg)

        # Apply global standardization
        if self.apply_global_std and self.global_mean is not None and self.global_std is not None:
            eeg = standardize_global(eeg, self.global_mean, self.global_std)

        # Apply Gaussian noise
        if self.apply_noise and self.split == 'train':
            eeg = add_noise(eeg)

        eeg = eeg.T  # Shape: (channels, time)
        eeg = torch.tensor(eeg.copy(), dtype=torch.float32)  # .copy() to avoid negative strides

        if self.split != 'test':
            label = torch.tensor(row['label_enc'], dtype=torch.long)
            return eeg, label
        else:
            return eeg