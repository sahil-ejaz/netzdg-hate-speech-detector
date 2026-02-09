import re
import streamlit as st
import requests
import json
import string
import plotly.graph_objects as go
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

GERMAN_STOPWORDS = set(stopwords.words('german'))
STEMMER = SnowballStemmer('german')

BEDROCK_ENDPOINT = "https://ctwa92wg1b.execute-api.us-east-1.amazonaws.com/prod/invoke"
TEAM_ID = "Sahil"
API_TOKEN = "0YtA2WdtwCAFYqKL2kpagxIjTco9xK3HK6CaSxWCoyk"

MODEL_MAPPING = {
    "Anthropic Claude 3 Haiku": "us.anthropic.claude-3-haiku-20240307-v1:0",
    "Meta Llama 3.3 70B": "us.meta.llama3-3-70b-instruct-v1:0",
    "Amazon Nova Pro": "us.amazon.nova-pro-v1:0",
    "Amazon Nova Lite": "us.amazon.nova-lite-v1:0"
}
MODEL_OPTIONS = list(MODEL_MAPPING.keys())

SUGGESTED_EXAMPLES = [
    "Invasoren raus",
    "Afrikaner sind dreckige Ungeziefer",
    "Frauen gehören ins Puff oder die Küche"
]

GROUP_REFERENCE_DIRECT = ['ausländ', 'flüchtling', 'migrant', 'frau', 'muslim', 'jude', 'asylant', 'araber', 'türk', 'afrik', 'syrer', 'afghan', 'alban', 'tschetsch', 'chechen', 'tunes', 'rumän', 'rumaen', 'bulgar', 'polack', 'polen', 'polnisch', 'russ', 'ukrain', 'serb', 'kroat', 'bosniak', 'kosovar']
GROUP_REFERENCE_IMPLICIT = ['invasor', 'goldstück', 'nafri', 'kopftuch', 'schleier', 'facharbeit', 'fachkraft']

GENERALIZATION_FULL = ['alle', 'diese', 'die', 'jede', 'jeder', 'sämtliche']
GENERALIZATION_PARTIAL = ['viele', 'einige', 'manche', 'die meisten']

EXCLUSION_DIRECT = ['raus', 'abschieben', 'vernichten', 'töten', 'weg', 'verschwinden', 'entfernen', 'ausrotten', 'vergasen']
EXCLUSION_INDIRECT = ['schützen vor', 'verteidigen gegen', 'wehren gegen', 'abwehren']

DEVALUATION_STRONG = ['dreck', 'ungeziefer', 'parasit', 'verdreck', 'abschaum', 'gesindel', 'pack', 'vieh', 'tier', 'ratten', 'kakerlak', 'schmarotzer', 'sau', 'schwein', 'drecksau', 'schweine', 'viech', 'hund', 'ratt']
DEVALUATION_MILD = ['faul', 'kriminell', 'gefährlich', 'primitiv', 'dumm', 'minderwertig', 'nutzlos']

TARGET_DIRECT = ['du', 'ihr', 'euch', 'dich', 'dir']

NEGATION_TERMS = ['nicht', 'kein', 'keine', 'keiner', 'niemals', 'nie', 'sarkasmus', 'ironie', 'scherz', 'witz']

VIOLENCE_IMMEDIATE = ['erschiess', 'erschieß', 'töten', 'toeten', 'umbring', 'abstech', 'köpf', 'koepf', 'vergas', 'lynch', 'aufhäng', 'aufhaeng']

ROLE_REDUCTION = ['küch', 'kuech', 'herd', 'puff', 'bordell', 'putz']
ROLE_NORMATIVE = ['gehör', 'gehoer', 'muss', 'muess', 'soll', 'dürf nicht', 'duerf nicht']

DEATH_VERBS = ['verreck', 'stirb', 'sterb', 'krepier']
DEATH_WISH_PHRASES = ['soll sterben', 'soll verrecken', 'sollte sterben', 'sollte verrecken', 'möge sterben', 'möge verrecken']
WISH_MARKERS = ['hoffentlich', 'möge']
DEATH_WISH_PRONOUNS = ['er', 'sie', 'es', 'du', 'ihr', 'euch', 'dich', 'dir', 'ihm', 'ihn', 'ihnen']


def preprocess_text(text):
    text_lower = text.lower()
    text_clean = text_lower.translate(str.maketrans('', '', string.punctuation))
    all_tokens = text_clean.split()
    filtered_tokens = [t for t in all_tokens if t not in GERMAN_STOPWORDS]
    stemmed_tokens = [STEMMER.stem(t) for t in filtered_tokens]
    return all_tokens, filtered_tokens, stemmed_tokens, text_clean


def check_pattern_match(tokens, stemmed_tokens, patterns):
    for pattern in patterns:
        for token in tokens:
            if pattern in token:
                return True
        for stem in stemmed_tokens:
            if pattern in stem:
                return True
    return False


def check_exact_match(tokens, terms):
    for term in terms:
        if term in tokens:
            return True
    return False


def check_phrase_match(text, phrases):
    for phrase in phrases:
        if phrase in text:
            return True
    return False


def check_substring_match(text, substrings):
    for sub in substrings:
        if sub in text:
            return True
    return False


def calculate_dimension_scores(text):
    all_tokens, filtered_tokens, stemmed_tokens, processed_text = preprocess_text(text)
    
    violence_override = check_substring_match(processed_text, VIOLENCE_IMMEDIATE)
    death_wish = (check_substring_match(processed_text, DEATH_VERBS) and check_substring_match(processed_text, WISH_MARKERS)) or check_substring_match(processed_text, DEATH_WISH_PHRASES)
    
    group_score = 0
    if check_pattern_match(filtered_tokens, stemmed_tokens, GROUP_REFERENCE_DIRECT):
        group_score = 1.0
    elif check_pattern_match(filtered_tokens, stemmed_tokens, GROUP_REFERENCE_IMPLICIT):
        group_score = 0.5
    
    generalization_score = 0
    if group_score > 0:
        if check_exact_match(all_tokens, GENERALIZATION_FULL):
            generalization_score = 1.0
        elif check_exact_match(all_tokens, GENERALIZATION_PARTIAL):
            generalization_score = 0.5
        if generalization_score == 0 and check_exact_match(all_tokens, ['die', 'diese', 'alle', 'jeder', 'jede']):
            generalization_score = max(generalization_score, 0.5)
        if 'genug davon' in processed_text:
            generalization_score = 1.0
        # Check for plural verbs with groups (e.g., "Afrikaner sind" = generalization)
        if check_exact_match(all_tokens, ['sind', 'waren', 'werden', 'wurden']) and group_score > 0:
            generalization_score = max(generalization_score, 1.0)
    
    exclusion_score = 0
    if check_pattern_match(filtered_tokens, stemmed_tokens, EXCLUSION_DIRECT):
        exclusion_score = 1.0
    elif check_phrase_match(processed_text, EXCLUSION_INDIRECT):
        exclusion_score = 0.5
    
    if death_wish:
        exclusion_score = max(exclusion_score, 0.5)
    
    devaluation_score = 0
    if check_pattern_match(filtered_tokens, stemmed_tokens, DEVALUATION_STRONG):
        devaluation_score = 1.0
        # If strong devaluation is combined with group reference, it's extremely severe
        if group_score > 0:
            devaluation_score = 1.0  # Keep at max, but this combination is very serious
    elif check_pattern_match(filtered_tokens, stemmed_tokens, DEVALUATION_MILD):
        devaluation_score = 0.5
    
    if group_score > 0 and check_substring_match(processed_text, ROLE_REDUCTION):
        devaluation_score = max(devaluation_score, 0.5)
        if check_substring_match(processed_text, ROLE_NORMATIVE):
            devaluation_score = 1.0
    
    target_score = 0
    if check_exact_match(all_tokens, TARGET_DIRECT):
        target_score = 1.0
    if death_wish and check_exact_match(all_tokens, DEATH_WISH_PRONOUNS):
        target_score = 1.0
    
    context_modifier = 0
    has_hate_content = (group_score > 0 or generalization_score > 0 or exclusion_score > 0 or devaluation_score > 0 or target_score > 0)
    if has_hate_content and not violence_override and not death_wish:
        negation_count = sum(1 for t in all_tokens if t in NEGATION_TERMS)
        if negation_count > 0:
            context_modifier = max(-1, -0.5 * negation_count)
    
    return {
        'group_reference': group_score,
        'generalization': generalization_score,
        'exclusion_violence': exclusion_score,
        'devaluation': devaluation_score,
        'target_directness': target_score,
        'context_modifier': context_modifier,
        'violence_override': violence_override,
        'death_wish': death_wish
    }


def find_matched_terms(text, pattern_list, match_type='pattern'):
    text_lower = text.lower()
    text_clean = text_lower.translate(str.maketrans('', '', string.punctuation))
    tokens = text_clean.split()
    stemmed = [STEMMER.stem(t) for t in tokens]
    matched = []
    for pattern in pattern_list:
        if match_type == 'exact':
            if pattern in tokens:
                matched.append(pattern)
        elif match_type == 'phrase':
            if pattern in text_clean:
                matched.append(pattern)
        elif match_type == 'substring':
            if pattern in text_clean:
                matched.append(pattern)
        else:
            for i, token in enumerate(tokens):
                if pattern in token and token not in matched:
                    matched.append(token)
                    break
            else:
                for i, stem in enumerate(stemmed):
                    if pattern in stem and tokens[i] not in matched:
                        matched.append(tokens[i])
                        break
    return matched


def generate_explanations(text, dimension_scores, total_score):
    explanations = []
    weights = {
        'group_reference': 1.5,
        'generalization': 1.0,
        'exclusion_violence': 2.0,
        'devaluation': 1.5,
        'target_directness': 1.0,
        'context_modifier': 1.0
    }

    violence_override = dimension_scores.get('violence_override', False)
    death_wish = dimension_scores.get('death_wish', False)

    group_val = dimension_scores['group_reference']
    group_matched = []
    if group_val > 0:
        group_matched = find_matched_terms(text, GROUP_REFERENCE_DIRECT, 'pattern')
        if not group_matched:
            group_matched = find_matched_terms(text, GROUP_REFERENCE_IMPLICIT, 'pattern')
    if group_val > 0:
        explanation = f"Group Reference was triggered because the text mentions a protected group (words like '{', '.join(group_matched)}'), which indicates the text is directed at a specific community."
    else:
        explanation = "Group Reference was not triggered because no protected group was mentioned in the text."
    contribution = group_val * weights['group_reference']
    explanations.append({
        'dimension': 'Group Reference',
        'triggered': group_val > 0,
        'explanation_text': explanation,
        'matched_terms': group_matched,
        'contribution': contribution
    })

    gen_val = dimension_scores['generalization']
    gen_matched = []
    if gen_val > 0:
        gen_matched = find_matched_terms(text, GENERALIZATION_FULL, 'exact')
        if not gen_matched:
            gen_matched = find_matched_terms(text, GENERALIZATION_PARTIAL, 'exact')
        phrase_matched = find_matched_terms(text, ['genug davon'], 'phrase')
        gen_matched = gen_matched + phrase_matched
    if gen_val > 0:
        explanation = f"Generalization was triggered because the text uses words like '{', '.join(gen_matched)}', which apply a statement broadly to an entire group rather than to individuals."
    else:
        if group_val == 0:
            explanation = "Generalization was not triggered because no group was referenced, so generalizing language was not considered."
        else:
            explanation = "Generalization was not triggered because the text does not use language that applies a statement to an entire group."
    contribution = gen_val * weights['generalization']
    explanations.append({
        'dimension': 'Generalization',
        'triggered': gen_val > 0,
        'explanation_text': explanation,
        'matched_terms': gen_matched,
        'contribution': contribution
    })

    excl_val = dimension_scores['exclusion_violence']
    excl_matched = []
    if excl_val > 0:
        excl_matched = find_matched_terms(text, EXCLUSION_DIRECT, 'pattern')
        if not excl_matched:
            excl_matched = find_matched_terms(text, EXCLUSION_INDIRECT, 'phrase')
        if death_wish:
            dw_matched = find_matched_terms(text, DEATH_VERBS, 'substring')
            dw_matched += find_matched_terms(text, DEATH_WISH_PHRASES, 'phrase')
            dw_matched += find_matched_terms(text, WISH_MARKERS, 'exact')
            excl_matched = excl_matched + [t for t in dw_matched if t not in excl_matched]
    if violence_override:
        vi_matched = find_matched_terms(text, VIOLENCE_IMMEDIATE, 'substring')
        excl_matched = excl_matched + [t for t in vi_matched if t not in excl_matched]
        explanation = f"Exclusion/Violence was triggered at the highest level because the text contains direct violent language ('{', '.join(excl_matched)}'), which calls for immediate action."
    elif excl_val > 0 and death_wish:
        explanation = f"Exclusion/Violence was triggered because the text expresses a death wish ('{', '.join(excl_matched)}'), which is a form of threatening language."
    elif excl_val > 0:
        explanation = f"Exclusion/Violence was triggered because the text contains language like '{', '.join(excl_matched)}', which expresses a call to exclude or remove people."
    else:
        explanation = "Exclusion/Violence was not triggered because the text does not contain exclusionary or violent language."
    contribution = excl_val * weights['exclusion_violence']
    explanations.append({
        'dimension': 'Exclusion/Violence',
        'triggered': excl_val > 0 or violence_override,
        'explanation_text': explanation,
        'matched_terms': excl_matched,
        'contribution': contribution
    })

    deval_val = dimension_scores['devaluation']
    deval_matched = []
    if deval_val > 0:
        deval_matched = find_matched_terms(text, DEVALUATION_STRONG, 'pattern')
        if not deval_matched:
            deval_matched = find_matched_terms(text, DEVALUATION_MILD, 'pattern')
        role_matched = find_matched_terms(text, ROLE_REDUCTION, 'substring')
        if role_matched:
            norm_matched = find_matched_terms(text, ROLE_NORMATIVE, 'substring')
            deval_matched = deval_matched + role_matched + norm_matched
    if deval_val > 0:
        explanation = f"Devaluation was triggered because the text uses words like '{', '.join(deval_matched)}', which dehumanize or diminish the targeted group."
    else:
        explanation = "Devaluation was not triggered because the text does not contain dehumanizing or degrading language."
    contribution = deval_val * weights['devaluation']
    explanations.append({
        'dimension': 'Devaluation',
        'triggered': deval_val > 0,
        'explanation_text': explanation,
        'matched_terms': deval_matched,
        'contribution': contribution
    })

    target_val = dimension_scores['target_directness']
    target_matched = []
    if target_val >= 1.0:
        target_matched = find_matched_terms(text, TARGET_DIRECT, 'exact')
        if not target_matched and death_wish:
            target_matched = find_matched_terms(text, DEATH_WISH_PRONOUNS, 'exact')
    if target_val >= 1.0:
        explanation = f"Target Directness was triggered because the text directly addresses someone using words like '{', '.join(target_matched)}', making the statement more personal and targeted."
    else:
        explanation = "Target Directness is at a low level because the text does not directly address a specific person, making the statement more diffuse."
    contribution = target_val * weights['target_directness']
    explanations.append({
        'dimension': 'Target Directness',
        'triggered': target_val >= 1.0,
        'explanation_text': explanation,
        'matched_terms': target_matched,
        'contribution': contribution
    })

    ctx_val = dimension_scores['context_modifier']
    ctx_matched = []
    if ctx_val < 0:
        ctx_matched = find_matched_terms(text, NEGATION_TERMS, 'exact')
    if ctx_val < 0:
        explanation = f"Context Modifier reduced the score because the text contains negating words like '{', '.join(ctx_matched)}', which may reverse or soften the hateful meaning of the statement."
    else:
        explanation = "Context Modifier did not adjust the score because no negation words were found that would reverse or soften a hateful statement."
    explanations.append({
        'dimension': 'Context Modifier',
        'triggered': ctx_val < 0,
        'explanation_text': explanation,
        'matched_terms': ctx_matched,
        'contribution': ctx_val
    })

    return explanations


def generate_overall_explanation(dimension_scores, total_score, recommendation):
    triggered_dims = []
    if dimension_scores['group_reference'] > 0:
        triggered_dims.append("references to a protected group")
    if dimension_scores['generalization'] > 0:
        triggered_dims.append("generalizing language")
    if dimension_scores['exclusion_violence'] > 0 or dimension_scores.get('violence_override', False):
        triggered_dims.append("exclusionary or violent language")
    if dimension_scores['devaluation'] > 0:
        triggered_dims.append("dehumanizing or degrading terms")
    if dimension_scores['target_directness'] >= 1.0:
        triggered_dims.append("direct targeting of a person")

    if total_score > 4.5:
        severity = "classified as high priority for removal"
    elif total_score > 3.0:
        severity = "classified as relevant under NetzDG for review"
    elif total_score >= 2.0:
        severity = "flagged for human review"
    else:
        severity = "classified as low risk"

    if not triggered_dims:
        return f"This text was {severity} because no significant hate speech indicators were found."

    if len(triggered_dims) == 1:
        reason_str = triggered_dims[0]
    elif len(triggered_dims) == 2:
        reason_str = f"{triggered_dims[0]} and {triggered_dims[1]}"
    else:
        reason_str = ', '.join(triggered_dims[:-1]) + f', and {triggered_dims[-1]}'

    return f"This text was {severity} because it contains {reason_str}."


def calculate_total_score(dimension_scores):
    if dimension_scores.get('violence_override', False):
        return 6.0
    total = (
        dimension_scores['group_reference'] * 1.5 +
        dimension_scores['generalization'] * 1.0 +
        dimension_scores['exclusion_violence'] * 2.0 +
        dimension_scores['devaluation'] * 1.5 +
        dimension_scores['target_directness'] * 1.0 +
        dimension_scores['context_modifier']
    )
    total = max(0, total)
    if dimension_scores.get('death_wish', False):
        total = max(total, 5.0)
    # Special case: Strong devaluation + group reference + generalization = very severe hate speech
    if (dimension_scores['group_reference'] > 0 and 
        dimension_scores['devaluation'] >= 1.0 and 
        dimension_scores['generalization'] > 0):
        total = max(total, 4.5)  # At least "NetzDG Relevant" level
    return total


def get_netz_dg_recommendation(score):
    if score > 4.5:
        return ("High Priority: Quick Removal", "error")
    elif score > 3.0:
        return ("NetzDG Relevant: Review or Deletion", "warning")
    elif score >= 2.0:
        return ("Human in the Loop: Consult human oversight with reference to HCAI (Human Centered AI framework)", "info")
    else:
        return ("Low Risk: No immediate action required", "success")


def get_sdg_impact(dimension_scores):
    impacts = []
    if dimension_scores['group_reference'] > 0:
        impacts.append("Supports SDG 10.2 by detecting xenophobia and discrimination")
    if dimension_scores['devaluation'] > 0:
        impacts.append("Supports SDG 16.1 by identifying dehumanizing language")
    if dimension_scores['exclusion_violence'] > 0:
        impacts.append("Supports SDG 16.2 by detecting violent exclusionary speech")
    
    if not impacts:
        impacts.append("No specific SDG violations detected")
    
    return impacts


def call_bedrock_api(prompt, model_id, max_tokens=1024):
    headers = {
        "Content-Type": "application/json",
        "X-Team-ID": TEAM_ID,
        "X-API-Token": API_TOKEN
    }
    
    payload = {
        "participant_id": TEAM_ID,
        "api_token": API_TOKEN,
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(BEDROCK_ENDPOINT, headers=headers, json=payload, timeout=60)
        result = response.json()
        
        if 'error' in result:
            error_msg = result['error']
            if 'Quota exceeded' in error_msg:
                return "API error: Budget quota exceeded. Contact hackathon organizers."
            return f"API error: {error_msg[:100]}"
        
        if not response.ok:
            return f"API error: HTTP {response.status_code}"
        
        if 'content' in result:
            content = result['content']
            if isinstance(content, list) and len(content) > 0:
                return content[0].get('text', str(content))
            return str(content)
        elif 'response' in result:
            return result['response']
        elif 'body' in result:
            body = result['body']
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except:
                    return body
            if isinstance(body, dict):
                if 'content' in body:
                    content = body['content']
                    if isinstance(content, list) and len(content) > 0:
                        return content[0].get('text', str(content))
                    return str(content)
            return str(body)
        return str(result)
    except requests.exceptions.Timeout:
        return "API error: Request timed out"
    except requests.exceptions.RequestException as e:
        return f"API error: Connection failed"
    except Exception as e:
        return f"API error: {str(e)[:50]}"


def is_repetitive(text):
    """Check if text contains excessive repetition."""
    if not text or len(text) < 30:
        return False
    lower = text.lower()
    if 'english translation:' in lower or 'translation:' in lower.split('.', 1)[-1]:
        return True
    words = text.split()
    if len(words) > 20:
        for phrase_len in range(3, min(10, len(words) // 3 + 1)):
            phrase = ' '.join(words[:phrase_len]).lower()
            count = lower.count(phrase)
            if count >= 3:
                return True
    sentences = [s.strip() for s in text.replace('. ', '.\n').split('\n') if s.strip()]
    if len(sentences) >= 3:
        unique = set(s.lower() for s in sentences)
        if len(unique) < len(sentences) * 0.5:
            return True
    return False


def clean_repeated_text(text):
    """Extract first meaningful sentence from repetitive output."""
    if not text or len(text) < 20:
        return text
    first_sentence = text.split('.')[0].strip()
    if first_sentence and len(first_sentence) > 5:
        return first_sentence
    return text[:100]


REFUSAL_INDICATORS = [
    "i do not", "i cannot", "i won't", "i will not", "i'm not able",
    "not able to", "refuse", "not engage", "not translate", "not generate",
    "hate speech", "discriminatory", "harmful", "inappropriate"
]

FALLBACK_MODEL = "us.anthropic.claude-3-haiku-20240307-v1:0"


def extract_translation(raw):
    """Extract clean translation from potentially messy LLM output."""
    if not raw or raw.startswith("API error"):
        return raw
    lines = [l.strip() for l in raw.strip().split('\n') if l.strip()]
    clean_lines = []
    for line in lines:
        for prefix in ['English translation:', 'Translation:', 'Output:']:
            if line.lower().startswith(prefix.lower()):
                line = line[len(prefix):].strip()
        if line:
            clean_lines.append(line)
    if not clean_lines:
        return raw
    first = clean_lines[0]
    first = first.split('. English translation:')[0]
    first = first.split('. Translation:')[0]
    first = first.rstrip('.')
    if first.endswith(','):
        first = first.rstrip(',')
    return first


def translate_to_english(text, model_id):
    prompt = f"""Translate this German text to English. Output ONLY the translation, nothing else. No alternatives, no commentary, no repetition.

German: "{text}"

English:"""
    result = call_bedrock_api(prompt, model_id, max_tokens=60)

    if any(indicator in result.lower() for indicator in REFUSAL_INDICATORS):
        if model_id != FALLBACK_MODEL:
            result = call_bedrock_api(prompt, FALLBACK_MODEL, max_tokens=60)

    return extract_translation(result)


def get_llm_classification(text, model_id):
    prompt = f"""Classify this text. Output ONLY these 3 lines:
Xenophobia Score: 0 or 1
Misogyny Score: 0 or 1
Explanation: short reason

Text: "{text}"

Xenophobia Score:"""
    raw = call_bedrock_api(prompt, model_id, max_tokens=80)
    raw = raw.strip()
    if not raw.lower().startswith("xenophobia"):
        raw = "Xenophobia Score:" + raw
    return extract_classification(raw)


def extract_classification(raw):
    lines = raw.split('\n')
    found = {}
    for line in lines:
        stripped = line.strip()
        for key in ['Xenophobia Score', 'Misogyny Score', 'Explanation']:
            if re.match(rf'^{key}\s*:', stripped, re.IGNORECASE) and key not in found:
                found[key] = stripped
                break
    if found:
        result = []
        for key in ['Xenophobia Score', 'Misogyny Score', 'Explanation']:
            if key in found:
                result.append(found[key])
        return '\n'.join(result)
    return raw


def create_score_chart(dimension_scores, total_score):
    dimensions = ['Group Reference\n(x1.5)', 'Generalization\n(x1.0)', 'Exclusion/Violence\n(x2.0)', 
                  'Devaluation\n(x1.5)', 'Target Directness\n(x1.0)', 'Context Modifier']
    
    weighted_scores = [
        dimension_scores['group_reference'] * 1.5,
        dimension_scores['generalization'] * 1.0,
        dimension_scores['exclusion_violence'] * 2.0,
        dimension_scores['devaluation'] * 1.5,
        dimension_scores['target_directness'] * 1.0,
        dimension_scores['context_modifier']
    ]
    
    colors = ['#FF6B6B' if s > 0 else '#4ECDC4' for s in weighted_scores]
    colors[-1] = '#95A5A6' if dimension_scores['context_modifier'] == 0 else '#3498DB'
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Weighted Score',
        x=dimensions,
        y=weighted_scores,
        marker_color=colors,
        text=[f'{s:.2f}' for s in weighted_scores],
        textposition='auto'
    ))
    
    fig.add_hline(y=total_score, line_dash="dash", line_color="red", 
                  annotation_text=f"Total: {total_score:.2f}", 
                  annotation_position="right")
    
    fig.update_layout(
        title='Hate Speech Dimension Scores',
        xaxis_title='Dimension',
        yaxis_title='Weighted Score',
        showlegend=False,
        height=400
    )
    
    return fig


def analyze_texts(texts, selected_model, compare_with_llm):
    results = []
    model_id = MODEL_MAPPING.get(selected_model, selected_model)
    for idx, text in enumerate(texts):
        if not text.strip():
            continue
        
        clean_text = text.strip()
        translation = translate_to_english(clean_text, model_id)
        dimension_scores = calculate_dimension_scores(clean_text)
        total_score = calculate_total_score(dimension_scores)
        recommendation, rec_type = get_netz_dg_recommendation(total_score)
        sdg_impacts = get_sdg_impact(dimension_scores)
        explanations = generate_explanations(clean_text, dimension_scores, total_score)
        overall_explanation = generate_overall_explanation(dimension_scores, total_score, recommendation)
        
        llm_result = None
        if compare_with_llm:
            llm_result = get_llm_classification(translation, model_id)
        
        results.append({
            'idx': idx + 1,
            'text': clean_text,
            'translation': translation,
            'dimension_scores': dimension_scores,
            'total_score': total_score,
            'recommendation': recommendation,
            'rec_type': rec_type,
            'sdg_impacts': sdg_impacts,
            'llm_result': llm_result,
            'explanations': explanations,
            'overall_explanation': overall_explanation,
            'model_name': selected_model
        })
    
    return results


def main():
    st.set_page_config(
        page_title="NetzDG Hate Speech Detector",
        page_icon="",
        layout="wide"
    )
    
    if 'view_mode' not in st.session_state:
        st.session_state['view_mode'] = 'input'
    if 'input_text' not in st.session_state:
        st.session_state['input_text'] = ""
    if 'results' not in st.session_state:
        st.session_state['results'] = []
    if 'selected_model' not in st.session_state:
        st.session_state['selected_model'] = MODEL_OPTIONS[0]
    elif st.session_state['selected_model'] not in MODEL_OPTIONS:
        st.session_state['selected_model'] = MODEL_OPTIONS[0]
    if 'compare_llm' not in st.session_state:
        st.session_state['compare_llm'] = False
    
    st.title("NetzDG Hate Speech Detector")
    
    if st.session_state['view_mode'] == 'input':
        st.markdown("""
        ### About This Tool
        
        This application analyzes German text for potential hate speech content in compliance with the 
        **Network Enforcement Act (NetzDG)**. It uses a sophisticated rule based scoring system to detect:
        
        **Xenophobia**: Language targeting ethnic, religious or national groups

        **Misogyny**: Language targeting or devaluing women

        The system evaluates text across multiple dimensions and provides actionable recommendations 
        aligned with NetzDG requirements and UN Sustainable Development Goals (SDGs).
        """)
        
        with st.sidebar:
            st.header("Suggested Examples")
            st.markdown("*Click to copy and paste into the text area:*")
            for example in SUGGESTED_EXAMPLES:
                st.code(example, language=None)
        
        st.subheader("Enter Text for Analysis")
        
        selected_model = st.selectbox(
            "Select LLM Model for Translation",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(st.session_state['selected_model']),
            key="model_select"
        )
        st.session_state['selected_model'] = selected_model
        
        compare_with_llm = st.checkbox("Compare with Bedrock LLM Classification", value=st.session_state['compare_llm'], key="compare_check")
        st.session_state['compare_llm'] = compare_with_llm
        
        user_text = st.text_area(
            "Enter text/post/comment (one per line for multiple texts)",
            value=st.session_state['input_text'],
            height=150,
            placeholder="Enter German text here...\nYou can enter multiple texts, one per line.",
            key="text_input_area"
        )
        st.session_state['input_text'] = user_text
        
        if st.button("Start Detection", type="primary", use_container_width=True):
            if user_text and user_text.strip():
                texts = [t.strip() for t in user_text.strip().split('\n') if t.strip()]
                with st.spinner("Analyzing texts..."):
                    results = analyze_texts(texts, selected_model, compare_with_llm)
                st.session_state['results'] = results
                st.session_state['view_mode'] = 'results'
                st.rerun()
            else:
                st.warning("Please enter some text to analyze.")
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray; font-size: 0.8em;'>
        NetzDG Hate Speech Detector | Rule based Classification System | 
        Supporting UN SDG 10 (Reduced Inequalities) and SDG 16 (Peace, Justice and Strong Institutions)
        </div>
        """, unsafe_allow_html=True)
    
    else:
        if st.button("Back to Input", type="secondary"):
            st.session_state['view_mode'] = 'input'
            st.rerun()
        
        st.header("Analysis Results")
        
        results = st.session_state.get('results', [])
        
        if not results:
            st.warning("No results to display.")
        else:
            for result in results:
                idx = result['idx']
                text = result['text']
                translation = result['translation']
                dimension_scores = result['dimension_scores']
                total_score = result['total_score']
                recommendation = result['recommendation']
                rec_type = result['rec_type']
                sdg_impacts = result['sdg_impacts']
                llm_result = result['llm_result']
                
                with st.expander(f"**Text {idx}**: {text[:50]}{'...' if len(text) > 50 else ''}", expanded=True):
                    col_left, col_right = st.columns([1, 1])
                    
                    with col_left:
                        st.markdown("**Original German Text:**")
                        st.info(text)
                        
                        st.markdown("**English Translation:**")
                        if "API error" in translation:
                            st.warning(translation)
                        else:
                            st.success(translation)
                    
                    with col_right:
                        st.markdown("**Dimension Scores:**")
                        score_data = {
                            "Group Reference": f"{dimension_scores['group_reference']:.2f} (x1.5 = {dimension_scores['group_reference']*1.5:.2f})",
                            "Generalization": f"{dimension_scores['generalization']:.2f} (x1.0 = {dimension_scores['generalization']*1.0:.2f})",
                            "Exclusion/Violence": f"{dimension_scores['exclusion_violence']:.2f} (x2.0 = {dimension_scores['exclusion_violence']*2.0:.2f})",
                            "Devaluation": f"{dimension_scores['devaluation']:.2f} (x1.5 = {dimension_scores['devaluation']*1.5:.2f})",
                            "Target Directness": f"{dimension_scores['target_directness']:.2f} (x1.0 = {dimension_scores['target_directness']*1.0:.2f})",
                            "Context Modifier": f"{dimension_scores['context_modifier']:.2f}"
                        }
                        
                        for dim, score in score_data.items():
                            st.markdown(f"  {dim}: {score}")
                        
                        st.markdown(f"**Total Score: {total_score:.2f}**")
                    
                    if rec_type == "error":
                        st.error(recommendation)
                    elif rec_type == "warning":
                        st.warning(recommendation)
                    elif rec_type == "info":
                        st.info(recommendation)
                    else:
                        st.success(recommendation)
                    
                    explanations = result.get('explanations', [])
                    overall_explanation = result.get('overall_explanation', '')
                    model_name = result.get('model_name', 'Unknown')
                    
                    st.markdown("---")
                    st.subheader("HEARTS inspired Explainability: Why was this text classified this way?")
                    
                    if overall_explanation:
                        overall_html = f"<div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-left: 4px solid #e94560; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px;'>"
                        overall_html += f"<p style='margin: 0; font-size: 15px; line-height: 1.5; color: #e0e0e0;'>{overall_explanation}</p></div>"
                        st.markdown(overall_html, unsafe_allow_html=True)
                    
                    for exp in explanations:
                        dim_name = exp['dimension']
                        triggered = exp['triggered']
                        exp_text = exp['explanation_text']
                        matched = exp['matched_terms']
                        contrib = exp['contribution']
                        
                        if triggered:
                            border_color = '#e94560'
                            status_badge = '<span style="background: #e94560; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;">TRIGGERED</span>'
                        else:
                            border_color = '#4a4a6a'
                            status_badge = '<span style="background: #4a4a6a; color: #aaa; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;">NOT TRIGGERED</span>'
                        
                        if dim_name == 'Context Modifier':
                            if contrib < 0:
                                points_text = f'{contrib:+.1f} points'
                                points_color = '#3498db'
                            else:
                                points_text = 'no change'
                                points_color = '#888'
                        elif triggered:
                            points_text = f'+{contrib:.1f} points'
                            points_color = '#e94560'
                        else:
                            points_text = '+0 points'
                            points_color = '#888'
                        
                        matched_html = ''
                        if matched:
                            terms = ' '.join([f'<span style="background: #2d2d4a; border: 1px solid #e94560; color: #ff6b6b; padding: 2px 8px; border-radius: 4px; font-family: monospace; font-size: 13px; margin-right: 4px;">{t}</span>' for t in matched])
                            matched_html = f'<div style="margin-top: 8px;"><span style="color: #999; font-size: 13px;">Matched words:</span> {terms}</div>'
                        
                        card_html = f"<div style='background: #0e1117; border-left: 3px solid {border_color}; padding: 14px 18px; border-radius: 6px; margin-bottom: 10px;'>"
                        card_html += f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>"
                        card_html += f"<span style='font-weight: 700; font-size: 15px; color: #fafafa;'>{dim_name}</span>"
                        card_html += f"<span>{status_badge}</span></div>"
                        card_html += f"<p style='margin: 0; color: #ccc; font-size: 14px; line-height: 1.5;'>{exp_text}</p>"
                        card_html += matched_html
                        card_html += f"<div style='margin-top: 8px;'><span style='color: #999; font-size: 13px;'>Contribution to score:</span> <span style='color: {points_color}; font-weight: 600; font-size: 14px;'>{points_text}</span></div>"
                        card_html += "</div>"
                        st.markdown(card_html, unsafe_allow_html=True)
                    
                    with st.expander("Understanding the Dimension Weights", expanded=False):
                        st.markdown("**Exclusion/Violence (x2.0)** has the highest weight because calls for violence or exclusion of people represent the most severe form of hate speech. This is also the most critical point under NetzDG, leading most quickly to a legal obligation for removal.")
                        st.markdown("**Group Reference (x1.5)** is weighted higher because targeting a protected group (ethnic, religious, or national) is a central characteristic of hate speech. Without a group reference, statements are often just general insults.")
                        st.markdown("**Devaluation (x1.5)** is also weighted higher because dehumanizing language (e.g. 'vermin', 'parasites') is historically particularly dangerous and directly violates human dignity.")
                        st.markdown("**Generalization (x1.0)** and **Target Directness (x1.0)** carry standard weight because they are amplifying factors. A generalization ('all foreigners') makes a statement worse but is not hate speech on its own. Similarly, direct address ('you') makes it more personal but is not punishable by itself.")
                        st.markdown("The weights ensure that the overall score reflects the actual severity of the statement. Calls for violence weigh more heavily than generalizations.")
                    
                    st.markdown("---")
                    st.markdown("**SDG Impact Assessment:**")
                    for impact in sdg_impacts:
                        st.markdown(f"  {impact}")
                    
                    fig = create_score_chart(dimension_scores, total_score)
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")
                    
                    if llm_result:
                        st.markdown("---")
                        st.markdown("**Bedrock LLM Comparison:**")
                        if "API error" in llm_result:
                            st.warning(llm_result)
                        else:
                            # Format so Xenophobia, Misogyny, Explanation appear line by line
                            formatted = re.sub(r'\s+Misogyny Score\s*:', r'\n\nMisogyny Score:', llm_result, flags=re.IGNORECASE)
                            formatted = re.sub(r'\s+Explanation\s*:', r'\n\nExplanation:', formatted, flags=re.IGNORECASE)
                            st.info(formatted)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray; font-size: 0.8em;'>
        NetzDG Hate Speech Detector | Rule based Classification System | 
        Supporting UN SDG 10 (Reduced Inequalities) and SDG 16 (Peace, Justice and Strong Institutions)
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()