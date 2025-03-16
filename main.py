from deeprpi.utils import RPIDataset

if __name__=="__main__":
    RPIDataset = RPIDataset(data_path='./data/NPInter5.csv',
                            batch_size=32,
                            num_workers=4,
                            rna_col='RNA_aa_code',
                            protein_col='target_aa_code',
                            label_col='Y',
                            padding=True,
                            rna_max_length=1000,
                            protein_max_length=1000,
                            truncation=False,
                            val_ratio=0.1,
                            test_ratio=0.1)
    RPIDataset.setup()
    train_dataloader = RPIDataset.train_dataloader()
    for i in train_dataloader:
        print(len(i))
        print('='*50)
        print(i[0])
        print('='*50)
        print(i[1])
        break