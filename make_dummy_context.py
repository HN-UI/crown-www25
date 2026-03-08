import os

data_mode = "small"
modes = ["train", "dev", "test"]

for mode in modes:
    in_path = f"../MIND-{data_mode}/{mode}/entity_embedding.vec"
    out_path = f"../MIND-{data_mode}/{mode}/context_embedding.vec"

    if not os.path.exists(in_path):
        print(f"[SKIP] missing: {in_path}")
        continue

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            terms = line.split("\t")
            entity = terms[0]

            # 최소한 entity + 100차원은 있어야 함
            if len(terms) < 101:
                raise ValueError(f"{in_path}: too few columns {len(terms)} for {entity} (need at least 101)")

            vec100 = terms[1:101]  # 앞 100차원만 사용
            fout.write(entity + "\t" + "\t".join(vec100) + "\n")
            n += 1

    print(f"[OK] {mode}: wrote {out_path} ({n} lines)")
