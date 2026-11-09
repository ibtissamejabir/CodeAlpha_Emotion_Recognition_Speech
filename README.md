# Emotion Recognition from Speech

Recognize human emotions (angry, disgust, fear, happy, neutral, sad, surprise) from short speech recordings using speech signal processing and deep learning.

Two models are trained and compared on the same data:

| | Model A | Model B |
|---|---|---|
| **Input** | MFCC features (+ delta) | Raw 16 kHz waveform |
| **Architecture** | CNN + bidirectional LSTM (trained from scratch) | Wav2Vec2-base transformer (pretrained, fine-tuned) |
| **Framework** | PyTorch | Hugging Face Transformers |
| **Purpose** | Classic baseline from the project brief | Stronger modern approach |

Everything lives in one notebook: `Emotion_Recognition_Speech.ipynb`.

---

## Project at a glance

```
Download 4 datasets
      |
      v
Unify labels into 7 emotions  ->  one table (path, emotion, dataset, speaker)
      |
      v
Remove duplicate files  ->  speaker-independent train / val / test split
      |
      +--> Model A: MFCC -> CNN -> BiLSTM -> 7 emotions
      |
      +--> Model B: waveform -> Wav2Vec2 -> classification head -> 7 emotions
      |
      v
Evaluate on unseen speakers: accuracy, macro-F1, confusion matrix
      |
      v
predict_emotion("your_file.wav")   +   save model
```

## Datasets

| Dataset | Speakers | Clips | Emotions | Notes |
|---|---|---|---|---|
| RAVDESS (speech) | 24 actors | 1,440 | neutral, calm, happy, sad, angry, fearful, disgust, surprised | "calm" is merged into "neutral" |
| TESS | 2 actresses | 2,800 | angry, disgust, fear, happy, neutral, sad, pleasant surprise | The Kaggle zip contains the data twice; the notebook removes the duplicates |
| CREMA-D | 91 actors | 7,442 | angry, disgust, fear, happy, neutral, sad | No "surprise" |
| SAVEE | 4 male speakers | 480 | angry, disgust, fear, happy, neutral, sad, surprise | |

All four are acted emotions recorded in controlled conditions. The notebook downloads them automatically (RAVDESS from Zenodo, the others from Kaggle).

## Requirements

- Google Colab with a **GPU runtime** (Runtime > Change runtime type > T4 GPU). The notebook was developed on a Tesla T4.
- Internet access for downloading datasets and the pretrained model.
- About 10 GB of free disk space (datasets, checkpoints).
- Python packages (installed by the first cell): `transformers`, `datasets`, `accelerate`, `evaluate`, `librosa`, `soundfile`, `torchaudio`. Colab already provides `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.

## How to run

1. Open `Emotion_Recognition_Speech.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Run **Runtime > Run all** (or run cells top to bottom).
4. Wait for training. Rough times on a T4:
   - Dataset download: a few minutes
   - MFCC extraction: 2-4 minutes
   - Model A training: a few minutes
   - Model B training: 40-60 minutes for 8 epochs (reduce `EPOCHS` in section 8 to go faster)
5. At the end, the notebook saves the final model as `emotion_wav2vec2_final.zip` and downloads it.

## Notebook sections

| # | Section | What it does |
|---|---|---|
| 1 | Setup | Installs libraries, imports, checks the GPU |
| 2 | Download | Fetches RAVDESS, TESS, CREMA-D, SAVEE |
| 3 | Inspect | Prints folder structures to verify the downloads |
| 4 | Unify | Parses file names into one table with 7 unified emotions |
| 5 | Label ids | Maps emotions to numbers 0-6 |
| 6 | Clean + explore | Removes duplicate TESS files, plots class balance, shows waveforms and MFCCs |
| 7 | Split | Speaker-independent 70/15/15 split, class weights |
| 8 | Model A | MFCC extraction and CNN-BiLSTM training |
| 9 | Model B | Wav2Vec2 fine-tuning |
| 10 | Evaluation | Reports, confusion matrices, per-dataset accuracy, training curves |
| 11 | Inference + save | `predict_emotion()` function and model export |
| 12 | Summary | Short summary and limitations for your report |

A line-by-line explanation of every cell is in [`CODE_EXPLAINED.md`](CODE_EXPLAINED.md).

## Evaluation protocol

- **Speaker-independent split.** Every speaker belongs to exactly one of train, validation or test. This prevents the model from "cheating" by recognizing voices, so the test score reflects how it would do on new people.
- **Metrics.** Accuracy and **macro-F1** (the average F1 across the 7 emotions, so the smaller "surprise" class counts as much as the others).
- **Class weights** compensate for "surprise" having fewer samples.
- The validation set picks the best checkpoint and triggers early stopping. The test set is used only once, at the end.

## Results

Part I, speaker-independent test set (1,552 clips from unseen speakers). Part II adds more rows, with confidence intervals, when you run it:

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| MFCC + CNN-BiLSTM | 0.548 | 0.543 |
| Wav2Vec2 (fine-tuned) | 0.692 | 0.688 |

Scores from speaker-independent splits are lower than those in many tutorials that split randomly. They are the more honest numbers.

## Using the trained model

After training, in the same notebook:

```python
emotion, probabilities = predict_emotion("/content/my_recording.wav")
print(emotion, probabilities)
```

To use the downloaded model elsewhere:

```python
import torch, librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

path = "emotion_wav2vec2_final"            # unzipped folder
model = Wav2Vec2ForSequenceClassification.from_pretrained(path).eval()
extractor = Wav2Vec2FeatureExtractor.from_pretrained(path)

wav, _ = librosa.load("clip.wav", sr=16000, duration=4.0)
inputs = extractor(wav, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
print(model.config.id2label[int(probs.argmax())], float(probs.max()))
```

Recordings should be 16 kHz-friendly speech, about 1-4 seconds, with one speaker and little background noise.

## Configuration you may want to change

| Setting | Where | Default | Effect |
|---|---|---|---|
| `EPOCHS` | Section 9 | 8 | Wav2Vec2 training length |
| `BATCH_SIZE`, `GRAD_ACCUM` | Section 9 | 16, 2 | Effective batch 32. Lower `BATCH_SIZE` if you run out of GPU memory |
| `LR` | Section 9 | 3e-5 | Fine-tuning learning rate |
| `MAX_SECONDS` | Section 9 | 4.0 | Longer clips are cropped to this length |
| `MODEL_NAME` | Section 9 | `facebook/wav2vec2-base` | Could be swapped for another Wav2Vec2-style model |
| `N_MFCC`, `DURATION` | Section 8 | 40, 3.0 s | MFCC feature size and clip length for Model A |
| `EPOCHS_A` | Section 8 | 40 | Maximum epochs for Model A (early stopping usually ends sooner) |
| `SEED` | Section 7 | 42 | Starting seed for the speaker split |

## Troubleshooting

- **"No GPU detected".** Change the runtime type to GPU and restart.
- **CUDA out of memory.** Set `BATCH_SIZE = 8` and `GRAD_ACCUM = 4` (same effective batch size).
- **Kaggle download is empty or `unzip` fails.** The Kaggle download URLs can require a login. Download the zips manually (or with the Kaggle CLI or `kagglehub`), upload them to `/content/datasets/`, and run the `unzip` lines.
- **A dataset shows 0 files.** Run section 3 and compare the printed folder layout with the glob patterns in section 4. Some Kaggle mirrors nest the files one folder deeper.
- **Colab disconnected during training.** Checkpoints are written to `/content/wav2vec2-emotion`, but Colab storage is wiped when the runtime resets. Mount Google Drive and change `output_dir` to a Drive folder if you want them to survive.
- **A library version error.** Transformers and datasets change quickly. If an argument name is rejected, check the library's release notes for the new name.

## Part II: advanced model (production-oriented)

The second half of the notebook (sections 13-23) upgrades the project from a student baseline to a more professional system. It reuses the same data and the same speaker split, so all numbers are comparable.

| Component | What it does |
|---|---|
| WavLM-base-plus backbone | Stronger pretrained speech model (any Hugging Face speech backbone works through one setting) |
| Weighted layer sum + attentive statistics pooling | Learns which transformer layers carry emotion and pools over time with attention, ignoring padding |
| Linear probe then fine-tune, layer-wise LR decay | More stable fine-tuning that keeps pretrained knowledge in lower layers |
| Augmentation | Random crop, gain, mild speed change, noise at random SNR, in-model time masking |
| Ensemble | Averages the new model(s) with Part I's Wav2Vec2; weight chosen on validation only |
| Speaker-level bootstrap | 95% confidence intervals and paired model comparisons (clips from one speaker are not independent) |
| Calibration + abstention | Temperature scaling, ECE, reliability diagram, accuracy-vs-coverage curve |
| Error analysis | Most frequent confusions, hardest and easiest speakers |
| Deployment package | `EmotionPredictor` (sliding windows for long audio, calibrated confidence, abstention), int8 benchmark, `MODEL_CARD.md`, Gradio demo |

Extra requirements: a GPU with about 12 GB or more (a T4 is enough for the default base model), about 3 GB of RAM for the cached audio, and about 40-60 minutes of training. For `wavlm-large` use a larger GPU or `bs=8, accum=4, grad_ckpt=True`.

Output files: `/content/artifacts/` (speaker split, dataset index, results, checkpoints) and `/content/emotion_advanced_deploy.zip` (weights, config, `meta.json`, `MODEL_CARD.md`).

Reload the deployed model in a new session by running the class definitions from sections 14-15 and the `EmotionPredictor` class from section 22, then:

```python
predictor = EmotionPredictor("emotion_advanced_deploy")
print(predictor.predict("clip.wav", threshold=0.5))
```

## Limitations

- The data is **acted** emotion in quiet rooms. Real conversations, noise, accents and spontaneous emotion are harder, so expect lower accuracy outside these datasets.
- TESS has only 2 speakers and SAVEE only male speakers, so the speaker mix is uneven. CREMA-D dominates the test set.
- "Surprise" is the smallest class and exists in only three datasets.
- "Calm" was merged into "neutral", so the two cannot be distinguished.
- Emotion labels are partly subjective, and some pairs (for example fear vs. surprise, or neutral vs. sad) are confused by humans too.
- Only the audio is used. Tone of voice is not the same as what a person actually feels, and this model should not be used to make decisions about people.
