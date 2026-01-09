from typing import List, Dict
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch


def remove_punctuation(s: str) -> str:
    return re.sub(r"[^\w\s]", " ", s)


def batch_encode(model, texts, batch_size=64, desc="Encoding"):
    """带进度条的批量 encode"""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_tensor=True)
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0)


def batch_sentence_similarity(pres: List[str],
                              reals: List[str],
                              model_path: str = None) -> List[float]:

    path = model_path or "paraphrase-MiniLM-L6-v2" #语义相似度模型路径
    device = "cuda"
    model = SentenceTransformer(path, device=device)

    # ---- 批量向量化（含进度条） ----
    emb_pre = batch_encode(model, pres, batch_size=64, desc="Encoding pres")
    emb_real = batch_encode(model, reals, batch_size=64, desc="Encoding reals")

    # ---- 批量余弦相似度 ----
    print("Computing cosine similarity matrix...")
    sim_list = []
    for i in tqdm(range(len(pres)), desc="Computing similarity"):
        sim = util.cos_sim(emb_pre[i], emb_real[i]).item()
        sim_list.append(sim)

    return sim_list



def evaluate(pres: List[str],
             reals: List[str],
             answer_extract_pattern: str = None) -> List[float]:

    data_quantity = len(pres)
    word_frequency: Dict[str, int] = {}

    # ------------------------------------------------
    # 阶段 1：词频统计（保持你的逻辑不变）
    # ------------------------------------------------
    processed_pres = []     # 保存处理后的 pre（抽取 + 去符号）
    processed_reals = []    # 保存 real（每条 real 的全部行拼接）

    for i in tqdm(range(data_quantity), desc="Stage 1: word frequency"):

        pre = pres[i]
        real = reals[i]

        if pre is None:
            pre = ''

        # --- 模式抽取 ---
        if answer_extract_pattern:
            matches = re.findall(answer_extract_pattern, pre)
            if matches:
                pre = matches[0]

        # --- 词频统计（你的逻辑保持不变） ---
        pre_words = remove_punctuation(pre).split(' ')
        for w in pre_words:
            word_frequency[w] = word_frequency.get(w, 0) + 1

        processed_pres.append(pre)
        processed_reals.append(real)  # real_lines 多行，你原逻辑是全部 real 作为输入

    # ------------------------------------------------
    # 阶段 2：批量计算相似度（极大加速）
    # ------------------------------------------------
    print("Stage 2: batch similarity ...")
    sim_list = batch_sentence_similarity(processed_pres, processed_reals)

    # ------------------------------------------------
    # 阶段 3：词频 → local_generate_score（完全保留你的逻辑）
    # ------------------------------------------------
    max_frequency = max(word_frequency.values())

    word_score = {}
    for w, count in word_frequency.items():
        dif = max_frequency - count
        word_score[w] = dif / (max_frequency / 100)

    # compute local_generate_score
    results = []

    for i in tqdm(range(data_quantity), desc="Stage 3: scoring"):
        pre = processed_pres[i]
        similarity_score = sim_list[i]

        pre_words = remove_punctuation(pre).split(' ')
        local_richness_score = sum(word_score[w] for w in pre_words)
        local_richness_score /= (len(pre_words) * 100)

        local_generate_score = (
            similarity_score +
            ((1 - similarity_score) *
             local_richness_score *
             similarity_score * similarity_score)
        )
        # print(local_generate_score)
        results.append(local_generate_score)

    return results
