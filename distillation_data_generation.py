from openai import OpenAI
import json
import time

# API 키 설정
#client = OpenAI(api_key = "")

# 카테고리 정의
theme = ["love", "betrayal", "desire", "illusion"]
tone = ["melancholic", "bitter", "pleading"]
language_style = ["archaic", "modern"]
poetic_device = ["metaphor", "personification", "antithesis"]
subject_focus = ["beloved", "soul", "self"]
conflict_type = ["inner_conflict", "unrequited_love"]
closure_type = ["twist", "reaffirmation", "moral"]

# 결과 임시 저장 버퍼
buffer = []
BATCH_SIZE = 50
OUTPUT_FILE = "gpt4o_distillation_sonnet_outputs.jsonl"
OUTPUT_FILE_2 = "Full_gpt4o_distillation_sonnet_outputs.txt"

# 전체 프롬프트 생성
results = []
for a in range(len(theme)):
    for b in range(len(tone)):
        for c in range(len(language_style)):
            for d in range(len(poetic_device)):
                for e in range(len(subject_focus)):
                    for f in range(len(conflict_type)):
                        for g in range(len(closure_type)):
                            seed = [a, b, c, d, e, f, g]
                            prompt = [
                                f"<theme: {theme[a]}>",
                                f"<tone: {tone[b]}>",
                                f"<language_style: {language_style[c]}>",
                                f"<poetic_device: {poetic_device[d]}>",
                                f"<subject_focus: {subject_focus[e]}>",
                                f"<conflict_type: {conflict_type[f]}>",
                                f"<closure_type: {closure_type[g]}>"
                            ]
                            results.append((seed, prompt))

# GPT-4o 호출 함수
def generate_sonnet_first_2_lines(client, prompt):
    system_prompt = "You are a poetic assistant. You generate Shakespearean sonnets in iambic pentameter with the correct rhyme scheme."
    full_prompt = "\n".join([
        "<task=Sonnet_Generation>",
        "<type=shakespearean>",
        "<meter=iambic_pentameter>",
        "<rhyme=ababcdcdefefgg>",
        *prompt,
        "Write only the **first two lines** of a Shakespearean sonnet based on the attributes above",
        "Do not explain. Do not include any headings. Just output the two lines of the poem.."
    ])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=100,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def generate_sonnet_full_lines(client, given_sonnet):
    first_two_lines = given_sonnet
    full_prompt = f"""<task=Sonnet_Generation>
    <style=shakespearean>
    <meter=iambic_pentameter>
    <rhyme_scheme=ababcdcdefefgg>
    {first_two_lines}
    
    Output only the poem."""

    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None
    

# flush 함수: 결과를 파일에 저장
def flush_to_file(data, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n\n".join(data) + "\n\n")  # 실제 줄바꿈 유지

# # index → output 매핑
index_to_sonnet = {}

with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        index_to_sonnet[item["index"]] = item["output"]

def format_sonnet_with_couplet_indent(sonnet_text: str) -> str:
    """
    14줄짜리 소네트를 받아 마지막 couplet(13, 14행) 앞에 띄어쓰기 2번 추가
    """
    lines = sonnet_text.strip().split("\n")
    if len(lines) == 14:
        # 마지막 2줄에 들여쓰기 추가
        lines[12] = "  " + lines[12]
        lines[13] = "  " + lines[13]
        return "\n".join(lines)
    return sonnet_text  # 줄 수가 다르면 그대로 반환



# 전체 처리 루프
for i, (seed, prompt) in enumerate(results):
    print(f"[{i+1}/{len(results)}] Processing seed {seed}...")
    
    #output = generate_sonnet_first_2_lines(client, prompt)
    given_sonnet = index_to_sonnet.get(i)
    if given_sonnet is None:
        print(f"Warning: No 2-line sonnet found for index {i}")
        continue  # skip or raise error
    output = generate_sonnet_full_lines(client, given_sonnet)
    output = json.loads(output) if isinstance(output, str) and output.startswith('"') else output
    output = format_sonnet_with_couplet_indent(output)
    
    # buffer.append({
    #     "index": i,
    #     "seed": seed,
    #     "prompt": prompt,
    #     "output": output
    # })
    
    # 원하는 포맷으로
    sonnet_text = f"{i}\n\n{output.strip()}"
    sonnet_text = sonnet_text.replace("\\n", "\n").strip()
    buffer.append(sonnet_text) 

    # BATCH_SIZE마다 저장
    if len(buffer) >= BATCH_SIZE:
        flush_to_file(buffer, OUTPUT_FILE_2)
        print(f"✓ Flushed {len(buffer)} items to {OUTPUT_FILE_2}")
        buffer = []  # 메모리 비움
        time.sleep(2)  # API 속도 제한 방지

# 마지막 남은 결과 저장
if buffer:
    flush_to_file(buffer, OUTPUT_FILE_2)
    print(f"✓ Flushed final {len(buffer)} items to {OUTPUT_FILE_2}")
