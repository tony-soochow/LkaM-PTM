The LkaM-PTM method constructs three sub-networks for the sequence, pre-trained, and structural modalities' representations: SeqNet, PLMNet, and StruNet, and finally uses an MLP for decoding and prediction.  

1.The datasets are stored in the "Datasets" folder.  
2.Run MG.py to obtain structural representations:
python Structure/main/MG.py  
3.Run main_MS.py for prediction:
python Sequence/main_MS.py
