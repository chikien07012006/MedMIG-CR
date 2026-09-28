# Báo cáo tuần: Baselines và Ablations

## 1. Baselines

### 1.1 Direct MLP

**Mục tiêu.** Xây dựng một direct disease ranker để tham khảo hiệu năng khi mô hình được giám sát trực tiếp trên query-target pairs, không thực hiện graph traversal hay reasoning qua path.

**Biến đổi dữ liệu và đầu vào.** Mỗi `seed_node_key` được ánh xạ sang node ID và embedding PrimeKG 64 chiều. Embeddings của các seed được tổng hợp bằng mean và max pooling rồi nối lại thành query vector 128 chiều. Mỗi query được ghép với các target disease đã map và 8 disease negatives lấy mẫu cho mỗi positive. Pathology không được đưa vào làm input.

**Kiến trúc và huấn luyện.** Query vector được chiếu từ 128 xuống 64 chiều. Với mỗi cặp query-disease, model ghép `[q, d, |q-d|, q*d]` thành vector 256 chiều, sau đó đi qua MLP `256 -> 256 -> 128 -> 1`, với ReLU và dropout 0.2. Model được huấn luyện bằng pairwise BPR logistic loss; checkpoint được chọn theo validation MRR. Candidate space là toàn bộ 22.205 disease nodes của PrimeKG.

**Kết quả full test.** Run `direct_retrieval_v1` đánh giá đủ 134.236 test queries; có 125.831/134.236 queries chứa target trong candidate set (candidate recall 0.9374).

| Metric | Kết quả |
|---|---:|
| MRR | 0.9087 |
| Recall@5 | 0.9363 |
| Recall@10 | 0.9370 |
| Recall@20 | 0.9372 |
| Recall@50 | 0.9374 |

**Diễn giải.** Kết quả cao một phần vì đây là supervised direct ranking: model học trực tiếp quan hệ giữa pooled seed embeddings và disease targets, sau đó chấm điểm toàn bộ candidate diseases. Nó không chịu giới hạn candidate reachability của graph traversal hoặc chất lượng path generation. Vì vậy đây là tham khảo về direct classification/ranking, không phải baseline tương đương với graph-retrieval pipeline chính; không nên dùng làm đối chứng chính để kết luận retrieval qua graph tốt hay kém.

**Lưu ý về một output khác.** `full_v2` báo MRR 0.9911 nhưng metadata ghi candidate space chỉ gồm 47 mapped targets, tức closed-set classification thay vì ranking trên 22.205 PrimeKG diseases. Không dùng con số này trong benchmark retrieval.

Artifacts: `Benchmark_baselines/Direct/MLP/outputs/direct_retrieval_v1/`.

### 1.2 Direct XGBoost

**Biến đổi dữ liệu.** Query được biểu diễn bằng cách nối mean và max của seed embeddings thành vector 128 chiều. Mỗi query-disease pair thành feature 320 chiều: query 128, disease embedding 64, `abs(query_mean - disease)` 64, và `query_mean * disease` 64. Tập disease candidates gồm đủ 22.205 disease nodes. Mỗi query group giữ positive targets và lấy 20 disease negatives; query không có positive trong candidate space bị bỏ khỏi training groups. Pathology và path features không được sử dụng.

**Model và inference.** Dùng `XGBRanker(objective="rank:pairwise")` với 200 estimators, depth 6 và learning rate 0.05 trong profile hiện có. Khi inference, model chấm điểm toàn bộ 22.205 diseases theo các chunk; không có graph traversal.

**Kết quả hiện có: validation profile, chưa phải full benchmark.** Profile chỉ dùng 5.000 train queries và 2.000 validation queries.

| Metric | Validation, 2.000 queries |
|---|---:|
| MRR | 0.5669 |
| Recall@5 | 0.9235 |
| Recall@10 | 0.9415 |
| Recall@20 | 0.9470 |
| Recall@50 | 0.9490 |
| Candidate recall | 0.9495 |

Chưa có kết quả full-train hoặc test trong output hiện tại; không đặt các số validation này cạnh test metrics như cùng protocol.

Artifacts: `Benchmark_baselines/Direct/XGBoost/outputs/profile_train5k_valid200/`.

### 1.3 Personalized PageRank (PPR)

**Tiền xử lý và truy vấn.** PrimeKG adjacency được hợp nhất với transpose để tạo graph vô hướng, tương tự quy ước mở rộng incoming/outgoing của beam search. Với mỗi query, các seed hợp lệ được ánh xạ sang graph IDs và nhận personalization mass đồng đều. PPR lặp trên graph; dangling mass được trả về seed distribution. Xếp hạng toàn bộ 22.205 disease candidates theo PageRank score. Đây là phương pháp không tham số, không dùng node embeddings hay path features.

**Thiết lập tuning.** Validation-only trên 2.000 query, tối đa 50 iterations, tolerance `1e-8`. Không có run test trong các output tuning hiện tại.

| Restart probability | MRR | Recall@5 | Recall@10 | Recall@20 | Recall@50 | Hội tụ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.0045 | 0.0035 | 0.0120 | 0.0140 | 0.0425 | 0/2.000 |
| 0.15 | 0.0046 | 0.0035 | 0.0135 | 0.0165 | 0.0425 | 0/2.000 |
| 0.20 | 0.0048 | 0.0055 | 0.0135 | 0.0165 | 0.0390 | 0/2.000 |

Restart 0.20 có MRR validation cao nhất trong ba cấu hình, nhưng cả ba đều chạm giới hạn 50 iterations và không query nào đạt tolerance. Đây là kết quả tuning sơ bộ; cần tăng iteration hoặc xử lý convergence trước khi chốt PPR baseline và chạy test.

Artifacts: `Benchmark_baselines/Graph_Retrieval/PPR/outputs/tune_restart_*_valid2000/`.

### 1.4 Relational GNN (R-GCN)

**Đồ thị và dữ liệu.** R-GCN dùng toàn bộ PrimeKG, gồm 82.240 nodes, 2.734.346 directed edges và 18 relation types. Node2vec vectors 64 chiều được dùng làm initial node features có thể huấn luyện. Train query lấy mean embedding cuối của seed nodes làm query representation; pathology không làm input. Candidate set là 22.205 PrimeKG disease nodes.

**Kiến trúc và huấn luyện.** Hai lớp `RGCNConv`: `64 -> 128` và `128 -> 64`, mỗi lớp dùng 9 bases. Query-disease scorer nhận `[q, d, q*d]` (192 chiều), qua MLP `192 -> 128 -> 1`. Huấn luyện bằng mean pairwise `softplus(-score_positive + score_negative)`, với 20 disease negatives trên mỗi positive; optimizer Adam, learning rate `1e-3`. Full graph được encode mỗi epoch; validation MRR chọn checkpoint tốt nhất.

**Kết quả hiện có.** Run dùng tối đa 200.000 train queries, 5.000 validation queries và 5.000 test queries; checkpoint tốt nhất ở epoch 10.

| Metric | Validation, 5.000 queries | Test, 5.000 queries |
|---|---:|---:|
| MRR | 0.0565 | 0.0558 |
| Recall@5 | - | 0.0542 |
| Recall@10 | 0.1038 | 0.1030 |
| Recall@20 | - | 0.1178 |
| Recall@50 | - | 0.2696 |
| Candidate recall | - | 0.9366 |

R-GCN học biểu diễn node qua relational message passing trên graph trong training; inference dùng embedding đã encode để xếp hạng candidates, không duyệt path riêng cho từng query.

Artifacts: `Benchmark_baselines/Relational_GNN/outputs/full_train_test5000/`.

## 2. Ablations

### 2.1 Seed-only + Beam Search (không MIND, không GRU)

**Thiết lập.** Lấy mean PrimeKG embedding của seed nodes làm query interest duy nhất rồi chạy semantic beam search trên graph. Beam mở rộng cả incoming và outgoing edges, tối đa 10 hops, width 128, 4.096 paths/interest, `alpha=1.0`, `beta=0.1`. Candidate collection dùng target universe cố định từ condition map; evaluator dùng target labels từng query. Đánh giá 5.000 test queries đầu, không có query nào bị bỏ do thiếu seed hoặc thiếu candidate.

| Metric | Kết quả |
|---|---:|
| MRR | 0.0843 |
| Recall@1 | 0.0158 |
| Recall@5 | 0.1484 |
| Recall@10 | 0.2490 |
| Recall@20 | 0.3738 |
| Recall@50 | 0.4106 |

Thời gian retrieval cộng dồn khoảng 839 giây (0.168 giây/query); tổng elapsed khoảng 856 giây. Đây là baseline để đo phần cải thiện khi thêm query encoder MIND.

Artifacts: `experiments/ablations/seed_only_beam/results/test5000/`.

### 2.2 InfoNCE MIND K=1 + Beam

Hai cấu hình dùng cùng K=1 checkpoint, beam settings và 5.000 test queries. Metrics sẽ được điền sau khi các run hoàn tất.

| Cấu hình | MRR | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|---|---:|---:|---:|---:|---:|---:|
| K=1 + beam, không post-hoc GRU |  |  |  |  |  |  |
| K=1 + beam + post-hoc GRU reranker |  |  |  |  |  |  |

GRU ở dòng thứ hai chỉ rerank các candidate paths hoàn tất sau beam search; không dùng GRU để hướng dẫn/prune beam. Reranker được train từ path data sinh bằng K=1 MIND checkpoint.

Artifacts dự kiến: `experiments/ablations/infonce_k1_beam/results/test5000/` và `experiments/ablations/infonce_k1_gru_posthoc/results/test5000/`.

## 3. Phạm vi và lưu ý so sánh

- Seed-only, MLP full test, R-GCN test và XGBoost/PPR validation tuning hiện dùng split hoặc số lượng query khác nhau; chỉ so sánh trực tiếp khi chạy lại cùng cohort và candidate protocol.
- XGBoost hiện mới có validation profile trên 5.000 train/2.000 validation queries. PPR hiện chỉ có validation tuning và chưa hội tụ trong 50 iterations. Cả hai chưa có test metrics chính thức.
- Direct MLP/XGBoost chấm điểm toàn bộ disease candidate space mà không graph traversal; các kết quả này cung cấp upper-reference cho direct supervised ranking, nhưng khác cơ chế với graph retrieval.
- MLP `full_v2` dùng candidate space đóng chỉ gồm 47 mapped targets nên kết quả của run đó không đại diện cho open-candidate retrieval benchmark.
- Hai hàng K=1 được để trống có chủ ý; điền metrics sau khi hoàn tất retrieval và evaluation.
