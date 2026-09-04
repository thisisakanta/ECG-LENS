import torch
import pandas as pd
import numpy as np
from bert_tokenizer import load_list
from torch.utils.data import Dataset, DataLoader

class ImpressionsDataset(Dataset):
        """The dataset to contain report impressions and their labels."""
        
        def __init__(self, csv_path, list_path):
                """ Initialize the dataset object
                @param csv_path (string): path to the csv file containing labels
                @param list_path (string): path to the list of encoded impressions
                """
                self.df = pd.read_csv(csv_path)
                # Column subset for the diagnostic subclass segment. Replace with the
                # class list of the segment being trained; it must match
                # constants.CONDITIONS and the segment's YAML under configs/ecgbert_eval/.
                self.df=self.df[['anterior myocardial infarction',
'left anterior/posterior fascicular block',
'left ventricular hypertrophy',
'ST-T changes',
'inferior myocardial infarction',
'septal hypertrophy',
'complete right bundle branch block',
'Wolf-Parkinson-White syndrome',
'left atrial overload/enlargement',
'normal ECG',
'ischemic',
'AV block',
'right atrial overload/enlargement',
'lateral myocardial infarction',
'ischemic in inferior leads',
'ischemic in anterolateral leads',
'non-specific ST changes',
'complete left bundle branch block',
'incomplete left bundle branch block',
'incomplete right bundle branch block',
'posterior myocardial infarction',
'right ventricular hypertrophy',
'non-specific intraventricular conduction disturbance (block)'
]]
                self.df.replace(0, 2, inplace=True) #negative label is 2
                self.df.replace(-1, 3, inplace=True) #uncertain label is 3
                self.df.fillna(0, inplace=True) #blank label is 0
                self.encoded_imp = load_list(path=list_path)

        def __len__(self):
                """Compute the length of the dataset

                @return (int): size of the dataframe
                """
                return self.df.shape[0]

        def __getitem__(self, idx):
                """ Functionality to index into the dataset
                @param idx (int): Integer index into the dataset

                @return (dictionary): Has keys 'imp', 'label' and 'len'. The value of 'imp' is
                                      a LongTensor of an encoded impression. The value of 'label'
                                      is a LongTensor containing the labels and 'the value of
                                      'len' is an integer representing the length of imp's value
                """
                if torch.is_tensor(idx):
                        idx = idx.tolist()
                label = self.df.iloc[idx].to_numpy()
                label = torch.LongTensor(label)
                imp = self.encoded_imp[idx]
                imp = torch.LongTensor(imp)
                return {"imp": imp, "label": label, "len": imp.shape[0]}
