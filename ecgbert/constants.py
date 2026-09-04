NUM_EPOCHS = 8        #Number of epochs to train for
BATCH_SIZE = 18       #Change this depending on GPU memory
NUM_WORKERS = 2       #A value of 0 means the main process loads the data
LEARNING_RATE = 2e-5
LOG_EVERY = 200       #iterations after which to log status during training
VALID_NITER = 200   #iterations after which to evaluate model and possibly save (if dev performance is a new max)
PRETRAIN_PATH = "checkpoints/bert_pretrained"  #path to pretrained model, such as BlueBERT or BioBERT
PAD_IDX = 0           #padding index as required by the tokenizer 

# CONDITIONS is the class list of the diagnostic segment being trained. It must
# match the `conditions` list in that segment's YAML under configs/ecgbert_eval/
# and the column subset in datasets/impressions_dataset.py. Set below to the
# 23-class diagnostic subclass segment; swap it to train another segment
# (super=5, rhythm=12, form=19) and pass the matching num_classes to bert_labeler.
CONDITIONS=[
    'anterior myocardial infarction',
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

]
CLASS_MAPPING = {0: "Negative", 1: "Positive", 2: "Negative", 3: "Uncertain"}
