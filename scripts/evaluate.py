import spacy
import textstat
import numpy as np
import pandas as pd
from rouge import Rouge
from statistics import mean
from nltk.metrics import edit_distance
from nltk import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

nlp = spacy.load("en_core_web_lg")

def calculate_static_scores(text):

    words = word_tokenize(text)

    avg_word_length = np.mean([len(word) for word in words])
    avg_sentence_length = np.mean([len(sentence) for sentence in sent_tokenize(text)])
    num_syllables = textstat.syllable_count(text)
    num_complex_words = textstat.difficult_words(text)
    num_long_words = max(len(word) for word in words)
    num_unique_words = len(set(words))
    num_monosyllabic_words = textstat.monosyllabcount(text)
    num_polysyllabic_words = textstat.polysyllabcount(text)
    flesch_kincaid = textstat.flesch_kincaid_grade(text)
    gunning_fog_index = textstat.gunning_fog(text)
    dale_chall_score = textstat.dale_chall_readability_score(text)

    return {
        "Average Word Length": avg_word_length,
        "Average Sentence Length": avg_sentence_length,
        "Number of Syllables": num_syllables,
        "Number of Complex Words": num_complex_words,
        "Number of Long Words": num_long_words,
        "Number of Unique Words": num_unique_words,
        "Number of Monosyllabic Words": num_monosyllabic_words,
        "Number of Polysyllabic Words": num_polysyllabic_words,
        "Flesch-Kincaid Grade": flesch_kincaid,
        "Gunning Fog Index": gunning_fog_index,
        "Dale-Chall Readability Score": dale_chall_score
    }

    # print(f"Average Word Length: {avg_word_length:.3f}")
    # print(f"Average Sentence Length: {avg_sentence_length:.3f}")
    # print(f"Number of Syllables: {num_syllables}")
    # print(f"Number of Complex Words: {num_complex_words}")
    # print(f"Number of Long Words: {num_long_words}")
    # print(f"Number of Unique Words: {num_unique_words}")
    # print(f"Number of Monosyllabic Words: {num_monosyllabic_words}")
    # print(f"Number of Polysyllabic Words: {num_polysyllabic_words}")
    # print(f"Flesch-Kincaid Grade: {flesch_kincaid}")
    # print(f"Gunning Fog Index: {gunning_fog_index}")
    # print(f"Dale-Chall Readability Score: {dale_chall_score}")

# Get one paragraph from the original text and paraphrased text
def calculate_comparative_scores(original_text, paraphrased_text):
    try:
        bleu_score = corpus_bleu([[word_tokenize(original_text)]], [word_tokenize(paraphrased_text)])

        tfidf_vectorizer = TfidfVectorizer(min_df=1)
        tfidf_matrix = tfidf_vectorizer.fit_transform([original_text, paraphrased_text])
        tfidf_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

        wer = edit_distance(word_tokenize(original_text), word_tokenize(paraphrased_text))
        cer = edit_distance(original_text, paraphrased_text)

        count_vectorizer = CountVectorizer()
        count_matrix = count_vectorizer.fit_transform([original_text, paraphrased_text])
        cosine_sim = cosine_similarity(count_matrix[0], count_matrix[1])

        original_set = set(word_tokenize(original_text))
        paraphrased_set = set(word_tokenize(paraphrased_text))
        jaccard_similarity = len(original_set.intersection(paraphrased_set)) / len(original_set.union(paraphrased_set))

        original_doc = nlp(original_text)
        paraphrased_doc = nlp(paraphrased_text)
        spacy_similarity = original_doc.similarity(paraphrased_doc)

        return {
            "bleu_score": bleu_score,
            "tfidf_similarity": tfidf_similarity[0][0],
            "word_error_rate": wer,
            "character_error_rate": cer,
            "cosine_similarity": cosine_sim[0][0],
            "jaccard_similarity": jaccard_similarity,
            "spacy_cosine_similarity": spacy_similarity
        }
    except:
        return {
            "bleu_score": 0,
            "tfidf_similarity": 0,
            "word_error_rate": 0,
            "character_error_rate": 0,
            "cosine_similarity": 0,
            "jaccard_similarity": 0,
            "spacy_cosine_similarity": 0
        }

def compute_average_comparative_scores(original_full_text, paraphrased_full_text):
    # Split into size 10k characters size buckets
    original_paragraphs = [original_full_text[i:i + 10000] for i in range(0, len(original_full_text), 10000)]
    paraphrased_paragraphs = [paraphrased_full_text[i:i + 10000] for i in range(0, len(paraphrased_full_text), 10000)]

    comparative_scores = [calculate_comparative_scores(o, p) for o, p in zip(original_paragraphs, paraphrased_paragraphs)]
    average_comparative_scores = {key: mean([d[key] for d in comparative_scores]) for key in comparative_scores[0]}
    print("Average Comparative Scores:", {k: f"{v:.3f}" for k, v in average_comparative_scores.items()})
    return average_comparative_scores

# Open the original and paraphrased text files
with open("datasets/the_great_gatsby.txt", "r") as f:
    original_full_text = f.read()

with open("outputs/20231207-131637/sentences.txt", "r") as f:
    paraphrase_sent = f.read()

with open("outputs/20231207-131637/paragraphs.txt", "r") as f:
    paraphrase_para = f.read()

with open("outputs/20231207-000740/sentences_corrected.txt", "r") as f:
    paraphrase_sent_grammar_corrected = f.read()

with open("outputs/20231207-000740/paragraphs_corrected.txt", "r") as f:
    paraphrase_para_grammar_corrected = f.read()

# Calculate Static Scores
original_static_scores = calculate_static_scores(original_full_text)
static_score_sentence_gen = calculate_static_scores(paraphrase_sent)
static_score_paragraph_gen = calculate_static_scores(paraphrase_para)
static_score_sentence_gen_grammar_corrected = calculate_static_scores(paraphrase_sent_grammar_corrected)
static_score_paragraph_gen_grammar_corrected = calculate_static_scores(paraphrase_para_grammar_corrected)

# Calculate Comparative Scores
comparative_scores_sentence_gen = compute_average_comparative_scores(original_full_text, paraphrase_sent)
comparative_scores_paragraph_gen = compute_average_comparative_scores(original_full_text, paraphrase_para)
comparative_scores_sentence_gen_grammar_corrected = compute_average_comparative_scores(original_full_text, paraphrase_sent_grammar_corrected)
comparative_scores_paragraph_gen_grammar_corrected = compute_average_comparative_scores(original_full_text, paraphrase_para_grammar_corrected)

# Create a DataFrame
data = {
    'Score Type': ['Average Word Length', 'Average Sentence Length', 'Number of Syllables',
                   'Number of Complex Words', 'Number of Long Words', 'Number of Unique Words',
                   'Number of Monosyllabic Words', 'Number of Polysyllabic Words',
                   'Flesch-Kincaid Grade', 'Gunning Fog Index', 'Dale-Chall Readability Score',
                   'BLEU Score', 'TF-IDF Similarity', 'Word Error Rate', 'Character Error Rate',
                   'Cosine Similarity', 'Jaccard Similarity', 'SpaCy Cosine Similarity'],
    'Original Text': [original_static_scores[key] for key in original_static_scores] + [None] * 7,
    'Sentence-wise Generated': [static_score_sentence_gen[key] for key in static_score_sentence_gen] + [comparative_scores_sentence_gen[key] for key in comparative_scores_sentence_gen],
    'Paragraph-wise Generated': [static_score_paragraph_gen[key] for key in static_score_paragraph_gen] + [comparative_scores_paragraph_gen[key] for key in comparative_scores_paragraph_gen],
    'Sentence-wise Generated (Grammar Corrected)': [static_score_sentence_gen_grammar_corrected[key] for key in static_score_sentence_gen_grammar_corrected] + [comparative_scores_sentence_gen_grammar_corrected[key] for key in comparative_scores_sentence_gen_grammar_corrected],
    'Paragraph-wise Generated (Grammar Corrected)': [static_score_paragraph_gen_grammar_corrected[key] for key in static_score_paragraph_gen_grammar_corrected] + [comparative_scores_paragraph_gen_grammar_corrected[key] for key in comparative_scores_paragraph_gen_grammar_corrected]
}

df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
df.to_excel("output_scores_1.xlsx", index=False)