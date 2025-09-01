import argparse
from pathlib import Path
from collections import Counter
import re

# --- PDF text extraction ---
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# --- Language & NLP ---
import spacy
from spacy.cli import download as spacy_download

# Progress bar (opzionale)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# --------------------- Utilità ---------------------
_NLP_CACHE = {}

EXTRA_STOPWORDS = {
    "’", "“", "”", "‘", "–", "—", "…", "«", "»",
    "could", "also", "would", "one", "two", "three",
    "oltre", "comunque", "quindi", "circa", "nonché",
    "via", "già", "poco", "molto", "po",
}

DEFAULT_ALLOWED_POS = {"NOUN", "PROPN", "ADJ"}

HYPHENS_REGEX = re.compile(r"[-‐\-–—]+")
SLASHES_REGEX = re.compile(r"[\/]+")


def _ensure_spacy_model(model_name: str):
    if model_name in _NLP_CACHE:
        return _NLP_CACHE[model_name]
    try:
        nlp = spacy.load(model_name, disable=["ner"])  # più veloce, NER non serve
    except OSError:
        spacy_download(model_name)
        nlp = spacy.load(model_name, disable=["ner"]) 
    _NLP_CACHE[model_name] = nlp
    return nlp


def get_nlp_english():
    return _ensure_spacy_model("en_core_web_sm"), "en"


def normalize_raw_text(s: str) -> str:
    s = HYPHENS_REGEX.sub(" ", s)
    s = SLASHES_REGEX.sub(" ", s)
    return s


def is_acronym(tok_text: str) -> bool:
    return tok_text.isupper() and 2 <= len(tok_text) <= 5 and tok_text.isalpha()


def valid_token(tok, allowed_pos, custom_stops_lower):
    if tok.is_space or tok.is_punct or tok.is_quote or tok.like_num or tok.like_url or tok.like_email:
        return False
    if tok.is_stop:
        return False
    if tok.pos_ not in allowed_pos:
        return False
    lemma_low = tok.lemma_.lower()
    if lemma_low in custom_stops_lower:
        return False
    if len(lemma_low) < 3 and not is_acronym(tok.text):
        return False
    if any(ch.isdigit() for ch in tok.text):
        return False
    return True


def normalize_token(tok):
    if is_acronym(tok.text):
        return tok.text  # preserva acronimi (es. AI)
    return tok.lemma_.lower()


def extract_text_from_pdf(path: Path) -> str:
    if pdfminer_extract_text is not None:
        try:
            return pdfminer_extract_text(str(path)) or ""
        except Exception:
            pass
    if PyPDF2 is not None:
        try:
            text_parts = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception:
            pass
    return ""


# --------------------- Pipeline ---------------------

def process_text(text: str, allowed_pos, custom_stops_lower):
    text = normalize_raw_text(text)
    nlp, lang_code = get_nlp_english()
    for sw in EXTRA_STOPWORDS:
        nlp.vocab[sw].is_stop = True
    doc = nlp(text)
    tokens = []
    for tok in doc:
        if valid_token(tok, allowed_pos, custom_stops_lower):
            tokens.append(normalize_token(tok))
    return tokens, lang_code


def count_bigrams(tokens):
    bigrams = []
    for i in range(len(tokens) - 1):
        t1, t2 = tokens[i], tokens[i + 1]
        if t1 == t2:
            continue
        bigrams.append(f"{t1} {t2}")
    return Counter(bigrams)


# --------------------- Main ---------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estrai le keyword (unigrammi) e i bigrammi più frequenti da PDF in una cartella. "
            "Normalizza case, lemmatizza, rimuove stopword e POS non rilevanti. "
            "Suggerimento: se tutti i PDF sono in inglese, non serve rilevamento lingua: è già fisso su inglese."
        )
    )
    parser.add_argument("--folder", required=True, help="Percorso della cartella con PDF (anche di rete)")
    parser.add_argument("--top", type=int, default=100, help="Quanti elementi stampare a video (default: 100)")
    parser.add_argument("--include-verbs", action="store_true", help="Includi anche i verbi tra le keyword")
    parser.add_argument("--extra-stopwords", default="", help="File .txt con stopword extra (una per riga)")
    parser.add_argument("--out-unigrams", default="keywords_unigrams.csv", help="CSV output unigrammi")
    parser.add_argument("--out-bigrams", default="keywords_bigrams.csv", help="CSV output bigrammi")
    args = parser.parse_args()

    root = Path(args.folder)
    if not root.exists():
        raise SystemExit(f"Cartella non trovata: {root}")

    allowed_pos = set(DEFAULT_ALLOWED_POS)
    if args.include_verbs:
        allowed_pos.add("VERB")

    custom_stops_lower = {sw.lower() for sw in EXTRA_STOPWORDS}
    if args.extra_stopwords:
        p = Path(args.extra_stopwords)
        if p.exists():
            extra = [
                line.strip().lower()
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip()
            ]
            custom_stops_lower |= set(extra)

    files = [p for p in root.rglob("*.pdf") if p.is_file()]
    if not files:
        print("Nessun PDF trovato.")
        return

    unigram_counter = Counter()
    bigram_counter = Counter()

    for pdf_path in tqdm(files, desc="Processo PDF"):
        text = extract_text_from_pdf(pdf_path)
        if not text or len(text.strip()) < 10:
            continue
        try:
            tokens, _ = process_text(
                text,
                allowed_pos=allowed_pos,
                custom_stops_lower=custom_stops_lower,
            )
            if not tokens:
                continue
            unigram_counter.update(tokens)
            bigram_counter.update(count_bigrams(tokens))
        except Exception:
            continue

    import csv
    with open(args.out_unigrams, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token", "conteggio"])
        for token, cnt in unigram_counter.most_common():
            w.writerow([token, cnt])

    with open(args.out_bigrams, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bigram", "conteggio"])
        for bg, cnt in bigram_counter.most_common():
            w.writerow([bg, cnt])

    print(f"\nTop {args.top} UNIGRAMMI:")
    for token, cnt in unigram_counter.most_common(args.top):
        print(f"{token:30s} {cnt}")

    print(f"\nTop {args.top} BIGRAMMI:")
    for bg, cnt in bigram_counter.most_common(args.top):
        print(f"{bg:30s} {cnt}")

    print(f"\nFatto. CSV salvati come: {args.out_unigrams} e {args.out_bigrams}")


if __name__ == "__main__":
    main()
