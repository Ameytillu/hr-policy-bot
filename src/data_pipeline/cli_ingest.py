import argparse, os, json, re, hashlib

def clean_text(t: str) -> str:
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def markdown_to_text(md_content: str) -> str:
    # remove markdown headings, bullets, symbols
    text = re.sub(r'[#*_`>-]', ' ', md_content)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # remove links
    return clean_text(text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True, help="Input folder with raw markdown policies")
    parser.add_argument("--out", dest="out", required=True, help="Output folder for processed JSONL")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "corpus.jsonl")

    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for fn in os.listdir(args.inp):
            if not fn.lower().endswith(".md"):
                continue

            policy_id = os.path.splitext(fn)[0]
            with open(os.path.join(args.inp, fn), "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            text = markdown_to_text(text)
            # split into ~500-character chunks (safe for embeddings)
            chunks = re.findall(r'.{1,500}(?:\s+|$)', text)

            for i, chunk in enumerate(chunks):
                chunk = clean_text(chunk)
                if len(chunk) < 60:
                    continue
                rec = {
                    "id": hashlib.md5(f"{policy_id}-{i}".encode()).hexdigest(),
                    "policy_id": policy_id,
                    "section": f"sec-{i:02d}",
                    "text": chunk,
                    "region": "GLOBAL",
                    "effective_from": "2025-01-01",
                    "source": f"file://{fn}#sec-{i:02d}"
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"Processed {n} chunks from Markdown policies into {out_path}")

if __name__ == "__main__":
    main()
