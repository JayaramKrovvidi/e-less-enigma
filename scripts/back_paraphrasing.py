import csv
import re
import nltk
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration


MODEL = "eugenesiow/bart-paraphrase"

tokenizer = BartTokenizer.from_pretrained(MODEL)
model = BartForConditionalGeneration.from_pretrained(MODEL)
paraphraser = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def paraphrase_sentence(sentence):
    paraphrased_sentences = paraphraser(sentence, max_length=50, num_return_sequences=3)
    return [entry['generated_text'] for entry in paraphrased_sentences]


def process_text_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        text = infile.read()

    sentences = nltk.sent_tokenize(text)

    dataset = []

    for sentence in sentences:
        cleaned_line = clean_text(sentence)
        paraphrased_lines = paraphrase_sentence(cleaned_line)

        for paraphrased_line in paraphrased_lines:
            dataset.append({'input_text': cleaned_line, 'output_text': paraphrased_line})

    with open(output_file_path, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['input_text', 'output_text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for entry in dataset:
            writer.writerow(entry)


if __name__ == "__main__":
    input_file_path = 'datasets/gadsby.txt'
    output_file_path = 'datasets/gadsby-data.csv'

    process_text_file(input_file_path, output_file_path)
