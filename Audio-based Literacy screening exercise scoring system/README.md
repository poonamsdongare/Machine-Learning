<p align="center">
  <img src="Images/Cover.jpg" alt="Description" />
</p>

## Introduction:
<br> This project was developed as a part of the [Good night moon early literacy program competition](https://www.drivendata.org/competitions/298/)[1]. The competition aims to leverage machine learning techniques to evaluate literacy screening exercise audio recordings from children in kindergarten through third grade. The goal is to assist teachers in efficiently and accurately identifying students who require early literacy intervention. Addressing these challenges at the preschool level is crucial, as early literacy development strongly correlates with later academic success.

## Approach:
<br> In this project, we leverage state-of-the-art pretrained audio speech recognition models and fine-tune them for our dataset. Specifically, we evaluated OpenAI’s Whisper[2] and Facebook’s wav2vec2 [3] models from the Hugging Face library. A comparative analysis revealed that Whisper outperformed wav2vec2, leading us to adopt it for further experimentation.
<br> Our initial assessment achieved a log loss (performance metric) value at the 50th percentile. However, we quickly identified that fine-tuning alone was insufficient to achieve significant performance improvements. To address this, we designed a custom Audio-Text Concat model, integrating both audio and text-based features for enhanced recognition. Further details regarding this architecture will be discussed in the **Model Development** section.

## Environment: 
<br> Initially, the environment was configured on a local machine; however, due to high computational demands, we migrated the setup to Google Colab. Given the inference time constraints of the competition, we utilized a T4 GPU for inference and an **A100 GPU** for model training. We leveraged pretrained models from the Hugging Face library and conducted the final code submission within a **Docker** environment to ensure reproducibility and compliance with competition requirements.

## Dataset:
<br> For this competition, a custom dataset was curated, comprising over 38,000 anonymized audio recordings in .wav format, capturing students performing literacy screening exercises. These recordings were collected during the development and evaluation of the Reach Every Reader literacy screener. The literacy screener uses computer adaptive testing, meaning that all students in a given grade will not see the same exercises.

<br> There are four types of literacy exercises (task) in the challenge data. These include:
1. **Deletion:** Delete an identified portion of a word and say the target word after the omission, e.g., "birdhouse without bird"
2. **Blending:** Combine portions of a word and say the target word out loud, e.g., "foot - ball"
3. **Nonword repetition:** Repeat nonwords of increasing complexity, e.g., "guhchahdurnam"
4. **Sentence repetition:** Repeat sentences of varying length and complexities without changing the sentence’s structure or meaning, e.g., "The girls are playing in the park".

<br>Training dataset format: 

|filename | task | expected_text | grade | label|
|---------|------|---------------|-------|------|
|Name of the given audio file in wav format <br>(e.g. hgxrel.wav)|Type of tasks from which the sample was generated <br>(deletion/ non-word repetition/ blending/ sentence repetition)|Given transcript of the audio file|The grade of the student whose audio was recorded <br>(KG/1/2/3)|Label of the sample <br>(1 if the transcript matches the audio, 0 otherwise)|

<br> Note: The labels in this challenge are manually assigned scores by trained evaluators that indicate whether the student's response was correct or incorrect.

## Data Preparation:
- Exploratory data analysis revealed that Unique values of expected_text are very limited compared to the size of the dataset:
  |task|total count of samples|count of unique expected_text values|
  |-|-|-|
  |sentence repetition|9490|181|
  |deletion and blending|16016|156|
  |nonword repetition|12589|138|

- To reduce the training time and avoid overfitting, we reduce the training dataset by selecting 3 instances of every unique combination of task, expected_text, grade, and label for training dataset and 1 instance for validation dataset. This reduces the dataset size from 38095 to 2627 for training and 914 for validation.
- The longest token size for the words is 11 and for the sentences is 18. Set it for the model when training and turn on padding.
- Standardize the sampling rate of all audio files to 16kHz.

## ModelDevelopment
For this project, we employed two methodologies. The first involved fine-tuning existing pretrained automatic speech recognition (ASR) models, specifically Whisper and Wav2Vec2. The second approach focused on developing a custom audio-text concatenation model. Our initial evaluation indicated that Whisper outperformed Wav2Vec2, leading us to prioritize it for further experimentation. However, we observed that fine-tuning alone did not yield significant performance improvements. Consequently, we pursued the second approach, constructing a custom audio-text concatenation model to enhance performance.
Below are the details about the architecture of the custom audio text coacatenation model.

### Audio Text Concat Architecture  
- This custom model is designed to generate embeddings for both audio and text (i.e., transcripts). These embeddings are then integrated with task and grade values to construct a final comprehensive embedding.

- We utilize OpenAI’s Whisper, a general-purpose speech recognition model known for its robust performance on English words. Whisper is available in multiple model sizes, allowing us to select an appropriate complexity level for our dataset. To determine the optimal model size, we begin with the smallest variant, whisper-tiny.en, which consists of 39 million parameters. We assess its ability to overfit the training dataset; if it fails to demonstrate sufficient fitting capability, we scale up to a larger model. However, given the dataset’s sample size, we hypothesize that this model size should be adequate.

- For transcript embeddings, we prioritize phoneme-based representations over semantic meaning. For instance, the words old and gold should exhibit higher cosine similarity than gold and metal, as our primary focus is on pronunciation rather than meaning. To achieve this, we used <a href="https://pypi.org/project/phonetics/">phonetics</a> library to generate embeddings aligned with the expected textual representations.

- The categorical parameters task and grade are encoded accordingly.

- The final embedding is constructed by concatenating the audio and transcript embeddings with the task and grade values. This composite embedding is then fed into a classifier head consisting of a fully connected layer. The architecture of this layer, including its size and depth, is determined based on the complexity required to facilitate effective learning while mitigating overfitting. Additionally, dropout layers are incorporated as a regularization mechanism to enhance model generalization.
- The schematics of the architecture are as follows:

![Audio and Text concatenated architecture schematic](https://github.com/pnkalan/Goodnight-Moon-Hello-Early-Literacy-Screening/blob/main/audio_text_concat_arch.png)

## Experimentation and Results
We conducted experiments on both the fine-tuned Whisper model and the custom-built models. This section presents the results obtained from the evaluation of both models.
## a. Whisper End-to-End encoder-decoder transformer
Training dataset: 3 samples per unique combinations, label = 1, exclude non-word samples.
Validation datasets: 1 samples per unique combinations
1. Words (true positives): label = 1, exclude non-word samples
2. Words (false positives): label = 0, exclude non-word samples
3. Non-words (true positives): label = 1, only non-word samples
4. Non-words (false positives): label = 0, only non-word samples

![Learning curve](https://github.com/pnkalan/Goodnight-Moon-Hello-Early-Literacy-Screening/blob/main/learning_curve.png)

Hence, the model shows good performance if we exclude non-word tasks. We use the model trained till epoch 3 as it shows minimum weighted loss.

## b. Audio and text concatenation model
### i. Training setup
These are the values of the hyperparameter before tuning:
1. Whisper model: openai/whisper-tiny.en (smaller whisper model with 57M parameters)
2. Optimizer: AdamW
3. Learning rate: 5e-5
4. Learning rate scheduler: Linear
5. Loss criterion: CrossEntropyLoss
6. Dropout: 0 (This is kept at 0 to verify the learning curve. Can be reduced later once overfitting is detected.)
7. Layer sizes:
   * Audio embedding: 384 (Whisper output standard size)
   * Text embedding: 16 (Kept lower as the number of unique words is low)
   * FC layer: Number of layers = 2, Hidden layer size = 16 (Can be increased if required)

### ii. Hyperparameter tuning
0. Learning curve for the default hyperparameter values given above: The model quickly overfits to training data. L2 regularization and dropout required. Apply one at a time.
![Results from Tune 0](https://github.com/pnkalan/Goodnight-Moon-Hello-Early-Literacy-Screening/blob/main/plot_tune_0.png)
1. After adding L2 regularization penalty factor of 1e-4: Very slight improvement but not enough. Apply droput in next iteration.
![Results from Tune 1](https://github.com/pnkalan/Goodnight-Moon-Hello-Early-Literacy-Screening/blob/main/plot_tune_1.png)
2. After applying dropout of 0.3: Very slight improvement but not enough.
![Results from Tune 2](https://github.com/pnkalan/Goodnight-Moon-Hello-Early-Literacy-Screening/blob/main/plot_tune_2.png)
3. Using only words for training and validation dataset (non-word samples dropped): Further improvement in validation score, but not enough.
![Results from Tune 3](https://github.com/pnkalan/Goodnight-Moon-Hello-Early-Literacy-Screening/blob/main/plot_tune_3.png)
4. Remove grade and task inputs:
![Results from Tune 4](https://github.com/pnkalan/Goodnight-Moon-Hello-Early-Literacy-Screening/blob/main/plot_tune_4.png)
5. Reduce batch size from 128 to 64:
![Results from Tune 5](https://github.com/pnkalan/Goodnight-Moon-Hello-Early-Literacy-Screening/blob/main/plot_tune_5.png)

## References
[1] https://www.drivendata.org/competitions/298/literacy-screening/page/925/

[2] https://huggingface.co/openai/whisper-large-v3

[3]https://huggingface.co/docs/transformers/en/model_doc/wav2vec2
## Contact
Contact me at poonamsdongare04@gmail.com to access the complete paper and more information


