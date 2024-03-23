import os
import json
import spacy
import random
import textstat
import language_tool_python
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to('cuda')

nlp = spacy.load("en_core_web_lg")
tool = language_tool_python.LanguageTool('en-US')
similarity_checker = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()

# Tokens with 'e'
e_token_ids = [i for i in range(len(tokenizer)) if 'e' in tokenizer.decode([i]).lower()]

def generate_text(text):
    prompt = "paraphrase: " + text + " </s>"

    encoding = tokenizer.encode_plus(prompt, padding='max_length', return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=512,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=5,
        num_beams=5,
        bad_words_ids=[[i] for i in e_token_ids],
        suppress_tokens=e_token_ids,
        no_repeat_ngram_size=2
    )

    output_texts = [tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True) for output in outputs]
    grammar_corrected_texts = [correct_grammar(output) for output in output_texts]
    
    original_text_embedding = similarity_checker.encode(text, convert_to_tensor=True)
    text_embeddings = [similarity_checker.encode(text, convert_to_tensor=True) for text in grammar_corrected_texts]
    
    cosine_scores = [util.pytorch_cos_sim(original_text_embedding, text_embedding) for text_embedding in text_embeddings]
    readability_scores = [textstat.flesch_reading_ease(text) for text in grammar_corrected_texts]
    combined_scores = [cosine_score.item() + readability_score for cosine_score, readability_score in zip(cosine_scores, readability_scores)]
    return grammar_corrected_texts[combined_scores.index(max(combined_scores))]


def generate_sentences(input_text):
    output_text = ""

    for paragraph in input_text.split('\n\n'):
        for sentence in paragraph.split('.'):
            sentence = sentence.strip()

            if len(sentence) == 0:
                continue

            generated_sentence = generate_text(sentence)
            output_text += generated_sentence + ' '

        output_text += '\n\n'

    return output_text


def generate_paragraphs(input_text):
    output_text = ""

    for paragraph in input_text.split('\n\n'):
        paragraph = paragraph.strip()

        if len(paragraph) == 0:
            continue

        generated_paragraph = generate_text(paragraph)
        output_text += generated_paragraph + '\n\n'

    return output_text

def replace_nouns(text):
    doc = nlp(text)

    # Load noun replacements
    with open('datasets/noun_replacements.json', 'r') as f:
        replacements = json.load(f)

    mapping = {}

    for ent in doc.ents:
        value = ent.text
        replaced_values = list(mapping.values())
        if 'e' not in value.lower():
            continue

        if ent.label_ == "PERSON" and value not in mapping:
            filtered_persons = [person for person in replacements["persons"] if person not in replaced_values]
            random_person = random.choice(filtered_persons)
            mapping[value] = random_person
        elif ent.label_ == "ORG"  and value not in mapping:
            filtered_org = [org for org in replacements["org"] if org not in replaced_values]
            random_org = random.choice(filtered_org)
            mapping[value] = random_org
        elif ent.label_ == "GPE" or ent.label_ == "LOC"  and value not in mapping:
            filtered_loc = [loc for loc in replacements["locations"] if loc not in replaced_values]
            random_loc = random.choice(filtered_loc)
            mapping[value] = random_loc
    
    for replaced_value, replacement in mapping.items():
        text = text.replace(replaced_value, replacement)
    
    return text

def correct_grammar(text):
    # try:
    #     matches = tool.check(text)
    #     filtered_matches = [match for match in matches if len(match.replacements) > 0 and not any('e' in rep.lower() for rep in match.replacements)]
    #     corrected_text = language_tool_python.utils.correct(text, filtered_matches)
    #     return corrected_text
    # except:
    #     print(f"Error in correcting grammar for: {text}")
    #     return text
    return text

def generate_eless_novel():
    with open('datasets/the_great_gatsby.txt', 'r') as f:
        input_text = f.read()

        input_text = replace_nouns(input_text)
        sentence = generate_sentences(input_text)
        paragraphs = generate_paragraphs(input_text)

        # Create Directory
        date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.mkdir(f'outputs/{date_time}')

        with open(f'outputs/{date_time}/sentences.txt', 'w') as f:
            f.write(sentence)

        with open(f'outputs/{date_time}/parpagraphs.txt', 'w') as f:
            f.write(paragraphs)

        # sentence = correct_grammar(sentence)
        # paragraphs = correct_grammar(paragraphs)

        # with open(f'outputs/{date_time}/sentences_corrected.txt', 'w') as f:
        #     f.write(sentence)

        # with open(f'outputs/{date_time}/parpagraphs_corrected.txt', 'w') as f:
        #     f.write(paragraphs)

generate_eless_novel()
