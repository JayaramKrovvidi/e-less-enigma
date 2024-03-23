import json
import spacy

nlp = spacy.load("en_core_web_lg")

def trim(value):
    if value.endswith("'s"):
        value = value[:-2]
    return value.strip()
    
def generate_noun_replacements():
    with open('datasets/word_corpus.txt', 'r') as f:

        doc = nlp(f.read()[:1000000])

        replacements = {
            "persons": [],
            "org": [],
            "locations": [],
        }

        for ent in doc.ents:
            value = ent.text

            if 'e' in value.lower() or not value.isalnum() and not any(x in value for x in ['-', ' ', '.']):
                continue

            if ent.label_ == "PERSON" and value not in replacements["persons"]:
                replacements["persons"].append(trim(value))
            elif ent.label_ == "ORG" and value not in replacements["org"]:
                replacements["org"].append(trim(value))
            elif (ent.label_ == "GPE" or ent.label_ == "LOC") and (value not in replacements["locations"]):
                replacements["locations"].append(trim(value))

        final_replacements = {
            "persons": list(set(replacements["persons"])),
            "org": list(set(replacements["org"])),
            "locations": list(set(replacements["locations"]))
        }

        # save as json file
        with open('datasets/noun_replacements.json', 'w') as f:
            json.dump(final_replacements, f)

generate_noun_replacements()