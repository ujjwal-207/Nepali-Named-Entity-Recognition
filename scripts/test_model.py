
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pathlib import Path

def get_entities_from_tokens(tokens, predictions, id2label):
    entities = []
    current_entity = []
    current_label = None
    current_tokens = []
    
    for token, pred_id in zip(tokens, predictions):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        label = id2label[pred_id]
        
        if token.startswith('##'):
            if current_entity:
                current_tokens.append(token[2:])  # Remove ##
        elif label.startswith('B-'):
            if current_entity:
                entity_text = ''.join(current_tokens)
                entities.append({
                    'text': entity_text,
                    'type': current_label
                })
            current_label = label[2:]
            current_tokens = [token]
            current_entity = [token]
        elif label.startswith('I-'):
            if current_entity:
                current_tokens.append(token)
                current_entity.append(token)
        else:
            if current_entity:
                entity_text = ''.join(current_tokens)
                entities.append({
                    'text': entity_text,
                    'type': current_label
                })
            current_entity = []
            current_label = None
            current_tokens = []
    
    if current_entity:
        entity_text = ''.join(current_tokens)
        entities.append({
            'text': entity_text,
            'type': current_label
        })
    
    return entities


def test_model_clean():
    
    model_path = "models/baseline/bert-base-multilingual-cased/final_model"
    
    if not Path(model_path).exists():
        print(f" Model not found at: {model_path}")
        alternatives = [
            "models/baseline/xlm-roberta-base/final_model",
            "models/baseline/distilbert-base-multilingual-cased/final_model",
        ]
        for alt in alternatives:
            if Path(alt).exists():
                model_path = alt
                break
        else:
            print("No trained model found!")
            return
    
    print(f"Model: {model_path}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    
    id2label = model.config.id2label
    
    print("✓ Model loaded successfully!\n")
    
    test_sentences = [
        "राम काठमाडौंमा बस्छन् ।",
        "सीता र गीता पोखरा गएकी थिइन् ।",
        "अनिल नेपाल सरकारमा काम गर्छन् ।",
        "डा. रमेश शर्माले अनुसन्धान गरे ।",
        "बर्मेली भाषा मात्रै जान्ने अनिल की श्रीमती ।",
    ]
    
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"{i}. Input: {sentence}")
        
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        preds = predictions[0].tolist()
        
        entities = get_entities_from_tokens(tokens, preds, id2label)
        
        if entities:
            print("   Entities found:")
            for entity in entities:
                print(f" '{entity['text']}' → {entity['type']}")
        else:
            print("   No entities found")
        
        print()
    


if __name__ == "__main__":
    test_model_clean()
