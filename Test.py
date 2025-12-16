import os
import json
import torch
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


IMAGE_DIR = "/media/dev/big-disk/Namitha/passport-llm/images2"
OCR_JSON_PATH = "ocr_results1.json"
FINAL_JSON_PATH = "passport_results1.json"

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_DIR = "/media/dev/big-disk/Namitha/passport-llm/outputs/qwen_passport_extractor"

INSTRUCTION = INSTRUCTION = """
You are an information extraction system.

TASK:
Extract passport information from noisy OCR text.

OUTPUT RULES (VERY IMPORTANT):
- Return ONLY a single valid JSON object
- Do NOT add explanations, comments, or extra text
- Do NOT repeat the input
- Do NOT wrap JSON in markdown
- If a field is missing or unclear, set it to null
- Use ISO date format: YYYY-MM-DD
- Country codes must be 3-letter ICAO codes (e.g., ARE, USA, IND)

REQUIRED JSON SCHEMA:
{
  "passport_no": string | null,
  "first_name": string | null,
  "last_name": string | null,
  "middle_name": string | null,
  "dob": string | null,
  "gender": "M" | "F" | null,
  "issue_date": string | null,
  "expiry_date": string | null,
  "nationality": string | null,
  "place_of_birth": string | null,
  "country_code": string | null,
  "authority": string | null,
  "mrz": string | null,
  "filename": string | null
}

STRICT RULES:
- Names must be UPPERCASE
- Do NOT guess values
- If expiry_date < issue_date, still return as extracted
- MRZ must contain '<' characters if present

Now extract the information from the OCR text below.
"""


print("🔤 Loading PaddleOCR (same as training)...")
ocr = PaddleOCR(
    lang="en",
    use_angle_cls=True,
    det_db_thresh=0.3,
    det_db_box_thresh=0.5,
    rec_batch_num=6,
    max_text_length=50,
    use_gpu=False
)



print("🧠 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16
)

print("🔗 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model = model.merge_and_unload()
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token


def extract_json(text):
    try:
        s = text.find("{")
        e = text.rfind("}")
        if s == -1 or e == -1:
            return None
        return json.loads(text[s:e+1])
    except Exception:
        return None


def run_llm(ocr_text, filename):
    prompt = f"[INST] {INSTRUCTION}\n\n{ocr_text} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()

    parsed = extract_json(decoded)
    if parsed is None:
        return {
            "filename": filename,
            "error": "JSON parsing failed",
            "raw_output": decoded
        }

    parsed["filename"] = filename
    return parsed


def extract_text_from_paddle(result):
    """
    Robust extraction that works for:
    - PaddleOCR v2 (list of lists)
    - PaddleOCR v3 (list of dicts)
    """
    texts = []

    # v3 format: list[dict]
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        texts = result[0].get("rec_texts", [])

    # v2 format: list[list]
    elif isinstance(result, list):
        for page in result:
            for line in page:
                if len(line) >= 2:
                    texts.append(line[1][0])

    return [t for t in texts if t.strip()]



ocr_outputs = {}
final_outputs = []

for file in sorted(os.listdir(IMAGE_DIR)):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, file)
    print(f"\n📄 Processing: {file}")

    # OCR
    result = ocr.ocr(img_path)
    texts = extract_text_from_paddle(result)
    ocr_text = "\n".join(texts)

    ocr_outputs[file] = texts

    # LLM
    llm_result = run_llm(ocr_text, file)
    final_outputs.append(llm_result)


with open(OCR_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(ocr_outputs, f, indent=2, ensure_ascii=False)

with open(FINAL_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(final_outputs, f, indent=2, ensure_ascii=False)

print("\n✅ Done!")
print("📄 OCR saved to:", OCR_JSON_PATH)
print("🧠 LLM results saved to:", FINAL_JSON_PATH)
