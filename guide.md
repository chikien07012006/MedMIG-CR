# Hướng dẫn chạy baseline: PPR, Katz, Seed-only beam, R-GCN

Hướng dẫn này dành cho máy Mac (Apple M1 Pro, 10 core, 16GB RAM), chạy song song trên CPU. Mọi lệnh đều chạy từ **thư mục gốc của repo**.

---

## 0. Chuẩn bị

### 0.1 Protocol dùng chung (đã chốt)
| Vai trò | File |
|---|---|
| Train (R-GCN) | `data/processed/ddxplus_v2_subsets/train_50k.csv` |
| Tune / early stopping | `data/processed/ddxplus_v2_subsets/valid_5k.csv` |
| Test (số liệu đưa vào paper) | `data/processed/ddxplus_v2_subsets/test_10k.csv` |
| Không gian ứng viên | `closed` = 47 target đã map (mặc định); `open` = 22,205 bệnh trong PrimeKG (tuỳ chọn) |
| Evaluator | `scripts/evaluation/evaluate_ddxplus_retrieval.py`, được mọi script gọi tự động |

Nếu thư mục `data/processed/ddxplus_v2_subsets/` chưa có, tạo lại bằng lệnh dưới. Lệnh này luôn cho ra cùng các file (seed 42).
```bash
.venv/bin/python scripts/preprocess/build_query_subsets.py
```

### 0.2 Số process và an toàn cho máy
| Tình huống | `--num_workers` (PPR, Katz, Seed-only) / `--num_threads` (R-GCN) |
|---|---|
| Vẫn đang dùng máy | **6** (CPU dùng khoảng 60–70%, máy vẫn mượt) |
| Chạy qua đêm, không dùng máy | **8** |
| Không nên | 10: chỉ nhanh hơn 8 khoảng 6% mà đẩy CPU lên 100% |

- Mỗi worker dùng khoảng 0.35GB RAM; R-GCN dùng khoảng 3GB. Máy bạn thường chỉ còn trống 3–4GB, nên **tắt bớt ứng dụng nặng** (Chrome nhiều tab, Docker…) và **không chạy hai job nặng cùng lúc**.
- Với job dài, thêm `caffeinate -i` ở đầu lệnh để máy không ngủ, và cắm sạc. Ví dụ: `caffeinate -i .venv/bin/python ...`
- Kết quả **không phụ thuộc** vào số worker: đã kiểm tra, file output giống hệt từng byte giữa 1 và 6 worker.
- Muốn chạy thử nhanh trước khi chạy thật, thêm `--limit 200` (R-GCN dùng `--limit_train 2000 --limit_eval 300 --epochs 1`).

### 0.3 Output của mỗi run
```
<out_dir>/
  predictions.csv          # patient_index, candidate, score, rank (định dạng chung cho mọi phương pháp)
  run_summary.json         # cấu hình, thời gian, coverage, metric (R-GCN: training_summary.json)
  evaluation/summary.json  # MRR, Recall@1/5/10/20/50 từ evaluator chung
  evaluation/by_patient.csv
```
Quy ước tên `out_dir`: `Benchmark_baselines/<nhóm>/<phương pháp>/outputs/<valid5k|test10k>_<cấu hình>`.

---

## 1. Personalized PageRank (PPR)

Script: [run_ppr_baseline.py](Benchmark_baselines/Graph_Retrieval/PPR/run_ppr_baseline.py). Không cần train.

**Bước 1: tune `restart_probability` trên valid_5k** (mỗi cấu hình khoảng 2.7 phút với 6 worker, cả lưới khoảng 15–25 phút):
```bash
for A in 0.10 0.15 0.20 0.30 0.50; do
  .venv/bin/python Benchmark_baselines/Graph_Retrieval/PPR/run_ppr_baseline.py \
    --queries_csv data/processed/ddxplus_v2_subsets/valid_5k.csv \
    --out_dir Benchmark_baselines/Graph_Retrieval/PPR/outputs/valid5k_a${A} \
    --restart_probability $A --num_workers 6
done
# Tuỳ chọn: thử thêm --ppr_score degree (chia theo degree để giảm thiên lệch về hub)
```

**Bước 2: chạy test_10k với cấu hình tốt nhất** (khoảng 5–6 phút):
```bash
.venv/bin/python Benchmark_baselines/Graph_Retrieval/PPR/run_ppr_baseline.py \
  --queries_csv data/processed/ddxplus_v2_subsets/test_10k.csv \
  --out_dir Benchmark_baselines/Graph_Retrieval/PPR/outputs/test10k_best \
  --restart_probability <A_tốt_nhất> --num_workers 6
```
Kiểm tra `run_summary.json` → `ppr_convergence.converged_fraction` phải bằng 1.0. Lần đo thử trên valid_5k với a = 0.15: hội tụ 100%, trung bình 66 vòng lặp.

---

## 2. Truncated Katz

Script: [run_katz_baseline.py](Benchmark_baselines/Graph_Retrieval/Katz/run_katz_baseline.py). Không cần train, rất nhanh (5k query mất khoảng 10 giây).

**Bước 1: tune trên valid_5k** (32 cấu hình, khoảng 5–10 phút):
```bash
for N in sym rw; do for B in 0.3 0.5 0.7 0.9; do for L in 2 3 4 6; do
  .venv/bin/python Benchmark_baselines/Graph_Retrieval/Katz/run_katz_baseline.py \
    --queries_csv data/processed/ddxplus_v2_subsets/valid_5k.csv \
    --out_dir Benchmark_baselines/Graph_Retrieval/Katz/outputs/valid5k_${N}_b${B}_L${L} \
    --katz_normalization $N --katz_beta $B --katz_max_length $L --num_workers 6
done; done; done
```
(`--katz_normalization none` là đếm walk thô. Khi dùng nó, β phải rất nhỏ, khoảng 0.001–0.01, vì degree tối đa là 5492.)

**Bước 2: chạy test_10k** với bộ (N, B, L) tốt nhất, đổi `--queries_csv` thành `test_10k.csv` và `--out_dir` thành `.../outputs/test10k_best`.

---

## 3. Seed-only beam search (ablation)

Script: [run_seed_only_beam.py](Benchmark_baselines/Graph_Retrieval/SeedOnlyBeam/run_seed_only_beam.py). **Không tune**: giữ nguyên hop, beam, α, β như main model để so sánh có kiểm soát.
```bash
caffeinate -i .venv/bin/python Benchmark_baselines/Graph_Retrieval/SeedOnlyBeam/run_seed_only_beam.py \
  --queries_csv data/processed/ddxplus_v2_subsets/test_10k.csv \
  --out_dir Benchmark_baselines/Graph_Retrieval/SeedOnlyBeam/outputs/test10k \
  --max_hops 10 --beam_width 128 --alpha 1.0 --beta 0.1 --num_workers 6
```
Ước tính khoảng 6–10 phút. Nếu sau này main model đổi hop hoặc beam (theo kết quả sensitivity), chạy lại script này với đúng cấu hình đó.

Script cũ `experiments/ablations/seed_only_beam/run_seed_only_beam.py` được giữ lại để tham chiếu kết quả cũ. Bản mới cho output giống hệt nhưng chạy song song.

---

## 4. R-GCN

Script: [run_rgcn_baseline.py](Benchmark_baselines/Relational_GNN/run_rgcn_baseline.py). Một lệnh làm cả ba việc: train trên train_50k, early stopping trên valid_5k, rồi xếp hạng và đánh giá trên test_10k.

```bash
caffeinate -i .venv/bin/python Benchmark_baselines/Relational_GNN/run_rgcn_baseline.py \
  --out_dir Benchmark_baselines/Relational_GNN/outputs/train50k_test10k_closed \
  --epochs 20 --batch_size 512 --learning_rate 1e-3 --num_threads 6
```
- Mỗi epoch khoảng 98 bước × 3.7 giây ≈ **6 phút**. Tối đa 20 epoch ≈ 2 giờ, thường dừng sớm hơn (patience 3).
- Log mỗi epoch in `loss` và `valid_mrr`. Kết quả nằm trong `training_summary.json` và `evaluation/summary.json`.
- (Tuỳ chọn) Thử `--learning_rate 3e-4` hoặc `3e-3`. Chọn theo `best_valid_mrr`, **không chọn theo test**.

---

## 5. Gom kết quả để chọn cấu hình và lập bảng

Liệt kê các run (valid hoặc test), sắp theo MRR:
```bash
.venv/bin/python - <<'EOF'
import json, glob
rows = []
for path in glob.glob("Benchmark_baselines/**/outputs/*/evaluation/summary.json", recursive=True):
    m = json.load(open(path))
    rows.append((m["mrr"], m["recall@1"], m["recall@5"], m["recall@10"], m["num_evaluated"], path.split("/outputs/")[0].split("/")[-1] + "/" + path.split("/outputs/")[1].split("/")[0]))
for r in sorted(rows, reverse=True):
    print(f"MRR {r[0]:.4f}  R@1 {r[1]:.4f}  R@5 {r[2]:.4f}  R@10 {r[3]:.4f}  n={r[4]:<6} {r[5]}")
EOF
```

---

## 6. Lưu ý khi đọc và báo cáo kết quả

1. **R@20 và R@50 gần như vô nghĩa trong protocol closed.** Chỉ có 47 ứng viên, nên phương pháp nào chạm tới hết 47 (PPR, Katz, R-GCN) đều có R@50 = 1.0. Trong bảng chính nên báo cáo **MRR, R@1, R@5, R@10** và **candidate coverage**. R@20/R@50 đưa vào phụ lục.
2. **Mốc ngẫu nhiên:** xếp 47 ứng viên ngẫu nhiên cho MRR khoảng 0.094 với một target. Lần đo thử trên valid_5k: PPR 0.146 (a = 0.15), Katz 0.122 (cấu hình mặc định). Hai baseline cấu trúc thuần này chỉ nhỉnh hơn ngẫu nhiên một chút.
3. **Coverage khác nhau giữa hai nhóm phương pháp.**
   - PPR, Katz và R-GCN chấm điểm mọi ứng viên: coverage ≈ 100%.
   - Beam search (Seed-only, main model) chỉ trả về ứng viên mà nó thực sự chạm tới: coverage khoảng 33–60%. Query không chạm được target bị tính 0 ở mọi metric.

   Đây là khác biệt cơ chế cần nói rõ trong paper. Nó cũng là luận điểm cho cải tiến search về reachability.
4. **Mọi phương pháp dùng cùng đồ thị** (tất cả relation, kể cả `disease_phenotype_negative`, giống beam search). Nếu muốn thử bỏ relation âm tính cho PPR/Katz, dùng `--exclude_relations disease_phenotype_negative`, nhưng khi đó nên làm cùng thay đổi cho main model để giữ công bằng.
5. **Chỉ tune trên `valid_5k.csv`.** Test chỉ chạy **một lần** với cấu hình đã chọn.
6. **Kết quả cũ không dùng cho paper:** `PPR/outputs/tune_restart_*_valid2000` (bản cũ, chạy trên không gian open, chưa hội tụ) và `Relational_GNN/outputs/full_train_test5000`, `profile_1k` (bản cũ, mỗi epoch chỉ 1 bước optimizer).

---

## 7. Thứ tự chạy đề xuất và tổng thời gian
| Thứ tự | Việc | Thời gian (6 worker) |
|---|---|---|
| 1 | Katz: tune trên valid và chạy test | khoảng 10 phút |
| 2 | PPR: tune trên valid và chạy test | khoảng 25–30 phút |
| 3 | Seed-only trên test_10k | khoảng 10 phút |
| 4 | R-GCN (nên chạy qua đêm hoặc lúc rảnh) | khoảng 1–2 giờ |

## 8. Gỡ lỗi nhanh
- **Máy chậm hoặc swap tăng:** giảm `--num_workers` (hoặc `--num_threads`) xuống 4.
- **Dừng một job:** `Ctrl+C`. Nếu còn process con, dùng `pkill -f run_ppr_baseline` (đổi tên script cho phù hợp).
- **Đo tài nguyên một run** (CPU, RAM, thời gian, dung lượng output):
  ```bash
  .venv/bin/python scripts/profiling/profile_command.py --label ppr_test --report_json /tmp/ppr_profile.json -- \
    .venv/bin/python Benchmark_baselines/Graph_Retrieval/PPR/run_ppr_baseline.py --out_dir /tmp/ppr_test --limit 500
  ```
