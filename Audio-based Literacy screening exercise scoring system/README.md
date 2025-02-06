## Introduction:
<br> This project was developed as a part of the [Good night moon early literacy program competition](https://www.drivendata.org/competitions/298/). The competition aims to leverage machine learning techniques to evaluate literacy screening exercise audio recordings from children in kindergarten through third grade. The goal is to assist teachers in efficiently and accurately identifying students who require early literacy intervention. Addressing these challenges at the preschool level is crucial, as early literacy development strongly correlates with later academic success.

## Approach:
<br> In this project, we leverage state-of-the-art pretrained audio speech recognition models and fine-tune them for our dataset. Specifically, we evaluated OpenAI’s Whisper and Facebook’s wav2vec2 models from the Hugging Face library. A comparative analysis revealed that Whisper outperformed wav2vec2, leading us to adopt it for further experimentation.
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




