# Keyword Generation by Fine Tuning BART

<b>Description:</b> Made use of Open Source datasets like KPTimes, KPCrowd, Inspec etc to fine tune a BART model to generation keyword phrases.

Trained the model by batching complete dataset into individual batches of 2000 datapoints. This was to ensure GPU VRAM is not overused. <br><br>
Total number of data points = 2,98,311 <br>
Number of batches trained on = 8 <br>
Total number of data points used for training = 16,000 <br>
Best Rouge 1 score: 33.8 <br><br>

## Tech Stack:
* python
* NLTK
* pytorch
* pickle
* hugging face transformers
