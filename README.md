🔤 Transliteration Model & 🎶 Lyrics Generation using GPT-2
This repository contains two deep learning projects:

Transliteration Model using a sequence-to-sequence GRU-based encoder-decoder architecture implemented in PyTorch.

Lyrics Generation model fine-tuned from GPT-2 using song lyrics from artists such as Khalid and Lady Gaga.

🚀 Project 1: Transliteration using PyTorch
This project builds a transliteration system that converts source script words into a target script using a character-level sequence-to-sequence GRU-based architecture.

✅ Features
Custom data loader for TSV input (with word frequency).

Vocabulary building with special tokens (<pad>, <sos>, <eos>).

Encoder-decoder architecture using GRU.

Training and evaluation loop.

Sample inference/predictions from trained model.

📁 File Format
The transliteration dataset should be in TSV format:

php-template
Copy
Edit
target_word<TAB>source_word<TAB>frequency
🛠️ How to Use
Prepare dataset: Update the path to the TSV file in:

python
Copy
Edit
dev_path = "/content/hi.translit.sampled.dev.tsv"
Train the model: Uncomment the training loop in the code:

python
Copy
Edit
for epoch in range(10):
    loss = train(model, loader, optimizer, criterion)
    acc = evaluate_accuracy(model, loader, tgt_vocab)
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
Predict sample transliterations:

python
Copy
Edit
pred = predict(model, "namaste", src_vocab, tgt_vocab)
🎤 Project 2: Lyrics Generation using GPT-2
This project fine-tunes OpenAI's GPT-2 model on a custom lyrics dataset to generate new, stylistic song lyrics.

✅ Features
Combines and cleans lyrics from multiple CSVs.

Prepares text dataset using HuggingFace Datasets.

Fine-tunes GPT-2 on the cleaned lyrics.

Generates creative lyrics using the fine-tuned model.

📁 Dataset
Place your CSV files in the format:

csv
Copy
Edit
Title,Lyric
Song1,"These are the lyrics..."
Song2,"More lyrics..."
🛠️ How to Use
Install dependencies:

nginx
Copy
Edit
pip install datasets transformers
Prepare and clean the data: Your CSV files should be named and placed accordingly:

python
Copy
Edit
khalid_df = pd.read_csv('/content/Khalid.csv')
gaga_df = pd.read_csv('/content/LadyGaga.csv')
Train the model: The training uses HuggingFace’s Trainer:

python
Copy
Edit
trainer.train()
Generate lyrics:

python
Copy
Edit
print(generator("I remember those nights when", max_length=100)[0]["generated_text"])
🧠 Requirements
Python 3.8+

PyTorch

Transformers

Datasets

Pandas

NumPy

Install with:

bash
Copy
Edit
pip install torch transformers datasets pandas numpy
📌 Notes
The transliteration model uses character-level GRU layers without attention.

The lyrics generation model uses GPT-2 with padding and token truncation.

Training time and performance may vary depending on hardware.

