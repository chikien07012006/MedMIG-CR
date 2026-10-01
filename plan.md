# MedMIG-CR — Plan (bản nháp để thảo luận)

> Ngày: 2026-09-30. Chưa thay đổi gì trong repo ngoài file này. Mỗi mục đều cần bạn duyệt trước khi mình làm.

---

## 0. Hai phát hiện ảnh hưởng tới cả 4 việc

**0.1. MIND không hoàn toàn "không học trực tiếp".**
[train_mind_infonce_ddxplus.py:293-320](src/medmigcr_mind/train_mind_infonce_ddxplus.py#L293-L320) dùng tập `target_nodes` (47 bệnh đích) làm candidate cho InfoNCE. Vì vậy MIND thực chất là một bộ phân loại có giám sát trên 47 lớp, còn khác biệt là nó xuất ra vector trong không gian đồ thị. Reviewer có thể dùng đúng lập luận mà ta đang dùng để loại MLP/XGBoost để phản biện lại.
→ Bắt buộc báo cáo thêm ablation **"MIND-direct"**: xếp hạng 47 target theo `max_k cos(interest_k, target)` và không search.
- Nếu MIND-direct ≥ pipeline thì beam search đang làm hại kết quả. Khi đó paper phải chuyển trọng tâm sang giá trị của path và khả năng giải thích, hoặc phải sửa search (mục 3).
- Nếu MIND-direct < pipeline thì đây là bằng chứng mạnh rằng graph search có đóng góp.

**0.2. Graph retrieval đang dùng không gian ứng viên đóng gồm 47 target.**
Beam nhận `target_node_ids` và endpoint bị lọc theo `target_universe` ([run_ddxplus_infonce_gru_rerank.py:404-466](scripts/retrieval/run_ddxplus_infonce_gru_rerank.py#L404-L466)). Để so sánh công bằng, mọi baseline phải chạy cùng protocol. Đề xuất báo cáo hai protocol:
- **P-closed**: ứng viên là 47 target đã map (protocol chính, khớp với pipeline hiện tại).
- **P-open**: ứng viên là 22,205 disease node (khó hơn, cho thấy khả năng tổng quát).

Kết quả đã kiểm tra: 5000 dòng đầu của test có phân bố pathology gần với toàn bộ tập test (total variation 0.035, đủ 49/49 pathology). Tập đánh giá hiện tại dùng được. Với bản paper cuối vẫn nên chạy trên toàn bộ 134k query, hoặc lấy mẫu phân tầng với seed cố định.

---

## 1. Baseline công bằng (cùng seed input → duyệt đồ thị → xếp hạng bệnh)

Tiêu chí: baseline nhận **cùng tập seed node**, dùng **cùng đồ thị PrimeKG** (và cùng node2vec nếu cần embedding), **tạo ứng viên bằng cách lan truyền hoặc duyệt đồ thị**, xếp hạng trong **cùng tập ứng viên**, và dùng cùng evaluator cùng 5000 (sau này 134k) query.

### Nhóm A — Không tham số / heuristic duyệt đồ thị (bắt buộc, rẻ)
| # | Baseline | Cách làm | Ghi chú |
|---|---|---|---|
| A1 | **PPR / Random Walk with Restart** (đã có) | Restart về các seed | Phải sửa phần hội tụ: tăng lên 200–500 iteration hoặc dùng power iteration có dung sai 1e-6 trên ma trận chuẩn hoá. Chạy lại ở cả P-closed và P-open. |
| A2 | **Đếm path / Katz bị cắt ở L hop** | score(d) = Σ_seed Σ_{ℓ≤L} β^ℓ·#paths(seed→d) (tính bằng nhân ma trận thưa) | Baseline "graph proximity" kinh điển, rất dễ giải thích |
| A3 | **Khoảng cách đường đi ngắn nhất / BFS nhiều nguồn** | Xếp theo số hop nhỏ nhất, hoà thì phá bằng số seed chạm tới | Cho biết target cách seed bao xa |
| A4 | **Seed-only beam** (đã có) | Beam không có MIND | Giữ lại làm ablation |

### Nhóm B — Học để lan truyền hoặc tìm path, điều kiện theo truy vấn (baseline chính của paper)
Đây là nhóm cùng "hệ bài toán" với MedMIG-CR: xuất phát từ các node nguồn, lan truyền hoặc chọn path có điều kiện theo truy vấn, rồi xếp hạng node đích.

| # | Baseline | Vì sao công bằng | Cách chuyển sang bài toán của ta |
|---|---|---|---|
| B1 | **NBFNet** (Zhu et al., NeurIPS 2021) | GNN dựa trên path kiểu Bellman–Ford, lan truyền từ node nguồn | Boundary condition = indicator trên **nhiều seed** (thay cho 1 head), query relation dùng một relation giả "diagnosis". Train bằng BCE/softmax trên target. Cần cắt subgraph k-hop quanh seed vì PrimeKG 2.7M cạnh sẽ nặng. |
| B2 | **A\*Net** (Zhu et al., NeurIPS 2023) — [code](https://github.com/DeepGraphLearning/AStarNet) | Học hàm ưu tiên để chọn top-K node/cạnh mỗi bước, tức là một phiên bản "beam search có học" của NBFNet. **Đối thủ trực tiếp nhất** của semantic beam | Giống B1, và scale được lên KG hàng triệu cạnh |
| B3 | **AdaProp** (Zhang et al., KDD 2023) — [code](https://github.com/LARS-research/AdaProp) | Lan truyền thích ứng có lấy mẫu tăng dần các node triển vọng, đúng bản chất "beam" | Khởi tạo từ nhiều seed |
| B4 | **RED-GNN** (Zhang & Yao, WWW 2022) | Relational digraph từ nguồn, mạnh ở tính giải thích bằng path | Tuỳ chọn, nếu còn thời gian |
| B5 | **MINERVA** (Das et al., ICLR 2018) / **MultiHopKG** (Lin et al., EMNLP 2018) | RL agent đi từng bước trên KG và trả về path, cùng ý tưởng "duyệt ra path" | Agent bắt đầu từ seed (mỗi seed một rollout, hoặc chọn seed ban đầu), reward = chạm target. Beam lúc suy luận giữ nguyên width/hop như ta. |
| B6 | **R-GCN** (đã có) | Học biểu diễn trên đồ thị | Giữ lại nhưng **chạy lại ở P-closed** để cùng protocol |

### Nhóm C — Truy xuất subgraph / path (KBQA và y khoa)
| # | Baseline | Ghi chú |
|---|---|---|
| C1 | **PCST subgraph retrieval** (kiểu G-Retriever, He et al. 2024) | Prize-collecting Steiner tree với prize = cosine(node, mean seed). Không cần học, rẻ. |
| C2 | **NSM / SR+NSM** (He et al. WSDM 2021; Zhang et al. ACL 2022) | Pipeline KBQA "topic entities → retrieve subgraph → reason", tương đồng về cấu trúc bài toán |
| C3 | **DR.KNOWS** (Gao et al., JMIR AI 2025) — [code](https://github.com/serenayj/DRKnows) | Multi-hop path retrieval từ khái niệm lâm sàng tới chẩn đoán, có path ranker dùng attention. **Rất sát với MedMIG-CR** (retriever + path ranker). Chỉ dùng phần retriever/ranker, bỏ phần LLM. |
| C4 | **Think-on-Graph** (ICLR 2024) hoặc LLM-guided beam | LLM làm hàm pruning cho beam. Tốn chi phí nên chỉ chạy trên 500–1000 query. Tuỳ chọn. |
| — | MedRAG (WWW 2025, [arXiv](https://arxiv.org/abs/2502.04413)) | Có đánh giá trên DDXPlus, nhưng dựa vào LLM + KG tự xây + truy xuất EHR, **không cùng hệ bài toán**. Chỉ nên nhắc trong related work. |

### Đề xuất tối thiểu cho paper
- **Bắt buộc**: A1 (PPR đã sửa), A2 (Katz), A4 (Seed-only), B1 (NBFNet) hoặc B2 (A\*Net), B5 (MINERVA), B6 (R-GCN ở P-closed), và ablation MIND-direct (mục 0.1).
- **Nên có**: B3 (AdaProp), C1 (PCST), C3 (DR.KNOWS retriever).
- **Bảng chính**: MRR, R@1/5/10/20/50, **candidate coverage** (tỉ lệ query mà target nằm trong tập ứng viên tìm được) và latency/query.

---

## 2. Xoá MLP và XGBoost

Chưa làm vì bạn dặn không đụng vào repo. Khi bạn duyệt, mình sẽ:
- `git rm -r Benchmark_baselines/Direct/` (gồm MLP, XGBoost và outputs; các file output lớn vốn đã bị gitignore nên cần xoá thêm trên đĩa).
- Sửa [README.md](README.md) (bảng "Baseline snapshot" và đoạn "Interpretation of direct ranking") và [report.md](report.md) mục 1.1–1.2.
- Gỡ `xgboost` khỏi [requirements.txt](requirements.txt).
- Làm trên branch riêng, commit riêng.

Lưu ý: nếu reviewer hỏi về upper bound, có thể thay MLP bằng "MIND-direct" (mục 0.1). Cách này nhất quán hơn vì dùng chính encoder của ta.

---

## 3. Ý tưởng cải thiện kết quả

### 3.1 Chẩn đoán trước khi sửa (làm đầu tiên, rẻ)
Số liệu hiện có cho thấy **nút thắt là khả năng tìm tới target, không phải khâu xếp hạng**:
- K=3 + GRU: R@20 = R@50 = 0.544, nên khoảng 46% query không bao giờ chạm được target.
- Data cho GRU (100k train query): chỉ 53,754 query (54%) có path dương.

Cần đo trên 5000 test query:
1. Khoảng cách hop ngắn nhất từ seed tới target (bằng BFS). Target có nằm trong 10 hop không?
2. Target bị prune ở hop nào (log thứ hạng của node trên path đúng tại mỗi hop).
3. Hop, relation và node type của các path đúng (có thể phần lớn là `phenotype —disease_phenotype_positive→ disease`, 1–2 hop).
4. Degree của target: nếu target có degree cao thì penalty `β·log(deg)` đang đẩy nó ra khỏi beam.
5. MIND-direct (mục 0.1).

### 3.2 Cải tiến search (xếp theo tỉ lệ lợi ích/chi phí)
1. **Chuẩn hoá điểm path theo độ dài.** Hiện tại path score là tổng các step (có thể âm vì có degree penalty), nên bị lệch theo số hop. Nên thử trung bình, `score_last_step`, hoặc `max cos` trên path.
2. **Gộp nhiều path về cùng một endpoint.** Hiện tại lấy `max` ([run_ddxplus_infonce_gru_rerank.py:143-156, 317-333](scripts/retrieval/run_ddxplus_infonce_gru_rerank.py#L143-L156)). Đổi sang `logsumexp` hoặc tổng có trọng số: một bệnh được nhiều seed hoặc nhiều interest chạm tới thì đáng tin hơn. Đây là cách "gom bằng chứng" giống PPR.
3. **Quota beam theo seed hoặc theo interest (diverse beam).** Tránh để beam bị dồn vào vùng của một seed có degree cao.
4. **Relation prior hoặc meta-path.** Học (hoặc đếm từ path dương trong train) xác suất `P(relation, node_type | hop)` rồi cộng vào step score. Có thể thay bằng ràng buộc cứng kiểu "chỉ đi qua phenotype/disease/gene".
5. **Điều chỉnh β / tắt degree penalty cho node loại disease**, tuning trên validation (không phải test).
6. **Heuristic kiểu A\*.** step score = cos(neighbor, interest) + λ·max_{d∈targets} cos(neighbor, d) (ước lượng "còn cách đích bao xa"). Vẫn không nhìn nhãn của từng query vì chỉ dùng tập 47 target cố định.
7. **Bằng chứng âm tính.** DDXPlus có các triệu chứng "không có". Có thể thêm chúng làm negative seed hoặc interest đẩy ra xa (−γ·cos).
8. **Trọng số seed từ routing của MIND.** Dùng coupling coefficient để ưu tiên seed khởi đầu thay vì xử lý các seed ngang nhau.

### 3.3 Cải tiến GRU reranker
1. **GRU hiện không nhận thông tin của truy vấn.** Token chỉ gồm relation, direction, node type và node ([gru_reranker.py](src/medmigcr_path_reranker/gru_reranker.py)), nên mọi bệnh nhân có cùng path sẽ nhận cùng điểm. Nên dùng interest vector (hoặc pooled seeds) làm hidden khởi tạo, hoặc concat interest vào head. Đây là thay đổi **có khả năng tăng MRR rõ nhất**.
2. **Tận dụng `use_score_features`** (beam score, độ dài path), vì code đã có sẵn.
3. **Gộp nhiều path cho mỗi endpoint** bằng attention/logsumexp thay vì lấy max.
4. **GRU-guided beam hiện chậm khoảng 100 lần** (25–44 s/query so với 0.44 s/query) và chưa cho kết quả vượt trội. Nếu giữ hướng này thì phải batch toàn bộ beam của một hop (và nhiều query) trong một lần forward, đồng thời train kiểu "learning to search" (early update, huấn luyện trên prefix) để tránh exposure bias.
5. **46% train query không có path dương** nên bị lãng phí. Có thể thêm path dương "oracle" (shortest path tới target) để GRU học được cả những path mà beam bỏ sót.

### 3.4 Luận điểm multi-interest
K=1 + GRU (MRR 0.497) đang cao hơn K=3 + GRU (0.459/0.474). K=3 chỉ hơn ở R@50 (0.604 so với 0.562). Gợi ý:
- Kết hợp hai điểm mạnh: dùng K=3 để tạo ứng viên (coverage cao), rồi dùng reranker có điều kiện truy vấn để xếp hạng.
- Thử gộp theo interest bằng trọng số attention thay vì lấy max.
- Phân tích theo pathology hoặc theo số seed: K>1 có giúp các ca có nhiều cụm triệu chứng không?

---

## 4. Server và estimate compute (trả lời mail của chị Linh)

### 4.1 Số liệu thực đo trên máy local (MacBook 10 core, 16GB)
| Tác vụ | Cấu hình | Thời gian |
|---|---|---|
| Retrieval K=1 | hop 10, beam 128 | **0.13–0.14 s/query** (5000 query mất 684–699 s) |
| Retrieval K=3 | hop 10, beam 128 | **0.44 s/query** (5000 query mất 2,251–2,271 s) |
| Retrieval K=3 + GRU-guided | hop 8, beam 64 | **25–44 s/query** |
| GRU rerank post-hoc | — | khoảng 1 ms/query (không đáng kể) |

| Data gen cho GRU (K=3, hop10, bw128, ≤512 path/query) | Dung lượng |
|---|---|
| 10k query | 139 MB |
| 100k query | **1.4 GB** (3.49M path âm, 163k path dương) |
| Toàn bộ 1.02M query (ngoại suy) | **khoảng 14 GB** jsonl |

### 4.2 Ngoại suy cho full data (1 process)
- Data gen K=3 trên 1.02M train: 1.02M × 0.44 s ≈ **125 giờ**.
- Test K=3 trên 134k query: khoảng **16 giờ**. Valid 132k: khoảng 16 giờ.
- Nếu cần thêm K=1, K=2 và các ablation, khối lượng còn nhân lên nữa.

### 4.3 Nhận định quan trọng: bottleneck là CPU, không phải GPU
- Beam search ([beam_search.py](src/medmigcr_kg/beam_search.py)) chạy vòng lặp Python trên từng beam item và từng neighbor, xử lý tuần tự từng query. Các script **không có multiprocessing/sharding**.
- Trên GPU, mỗi lần gọi `score_neighbors` chỉ là một tensor rất nhỏ, nên chi phí chép dữ liệu CPU↔GPU chiếm phần lớn. **Một chiếc 4090 gần như không làm retrieval nhanh hơn**, thậm chí có thể chậm hơn so với `--graph_device cpu`.
- GPU chỉ thực sự cần cho: train MIND (nhẹ), train GRU (nhẹ), các baseline GNN (NBFNet/A\*Net/R-GCN trên 2.7M cạnh, nên 24GB là vừa đủ hoặc thiếu) và GRU-guided beam nếu được batch lại.
- Vì vậy: **số core CPU quyết định thời gian**. Cần thêm tính năng `--shard_id/--num_shards` (hoặc multiprocessing pool) vào các script retrieval và data gen. Mỗi process chỉ cần khoảng 1–2 GB RAM (graph CSR + embeddings chỉ 39MB, cộng thêm model và các tập visited path).
- Với 32 process song song: data gen 1.02M ≈ **4–5 giờ**, test 134k ≈ **30–40 phút**. Như vậy thuê 3 ngày là thừa nếu có đủ core. Ngược lại, nếu instance chỉ có 8–16 vCPU thì 3 ngày có thể không đủ để chạy hết ablation.

### 4.4 Việc cần làm trước khi nhận server (trên branch riêng, cần bạn duyệt)
1. Thêm sharding (`--shard_id`, `--num_shards`, ghi output theo shard rồi merge) cho `build_gru_path_reranker_dataset.py`, `run_ddxplus_infonce_gru_rerank.py` và `run_ddxplus_infonce_retrieval.py`.
2. Viết script profiling `scripts/profiling/profile_run.sh` để ghi lại:
   - GPU: `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1`
   - CPU/RAM: `psutil` hoặc `vmstat 1` / `pidstat -r -u 1`
   - Runtime và peak RSS: `/usr/bin/time -v`
   - Throughput: query/s (lấy từ summary JSON)
   - Dung lượng: `du -sh` output
3. Kế hoạch profiling (2–3 giờ đầu trên server):
   - a) Retrieval K=3, 1000 test query: so sánh `--graph_device cpu` và `cuda` (để chứng minh GPU không giúp).
   - b) Data gen K=3, 2000 train query với 1 / 8 / 16 / 32 process: đo scaling.
   - c) Train GRU 1 epoch trên GPU: đo VRAM.
   - d) (Nếu đã có) NBFNet/A\*Net 1 epoch: đo peak VRAM. Đây là con số quyết định có cần GPU lớn hơn hay không.
4. Chuẩn bị dữ liệu upload khoảng 1.5 GB: `train/valid/test_queries.csv` (khoảng 600MB), `primekg_graph/` (39MB), mappings và checkpoint `.pt`. **Không cần** upload `kg_giant.csv` hay các file patients gốc.

### 4.5 Dung lượng lưu trữ
Dữ liệu đầu vào khoảng 2 GB, data gen full K=3 khoảng 14 GB, thêm K=1 khoảng 7 GB, predictions và checkpoint vài GB, cộng venv/CUDA khoảng 10 GB. Tổng **khoảng 40 GB**, nên **150 GB là dư**.

### 4.6 Nháp email trả lời (để bạn sửa rồi gửi)

> Hi Linh,
>
> Thanks a lot for the plan and the guide — 1× RTX 4090 / 150 GB / 3 days works for me, and I'll start with a profiling run on a small subset before any full experiment.
>
> One point from local profiling that affects the instance choice: my current bottleneck (beam-search retrieval and GRU training-data generation) is **CPU-bound** — it's a per-query graph traversal in Python, currently ~0.44 s/query single-process. The GPU is mainly needed for model training (MIND, GRU reranker, and GNN baselines). So throughput will scale with the number of CPU cores (I'm adding multi-process sharding).
>
> Additional requirements, if possible:
> - **CPU:** ≥ 32 vCPU (ideally 32 physical cores / 64 threads)
> - **RAM:** ≥ 64 GB (128 GB preferred for GNN baselines on the full PrimeKG)
> - **Storage:** 150 GB is sufficient (estimated usage ~40 GB)
> - **CUDA:** 12.1+ with a PyTorch 2.x image (e.g. `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`); I'll also install torch-geometric
> - **Network:** a stable connection for uploading ~2 GB of data
>
> For profiling I'll record: peak VRAM and GPU utilization (nvidia-smi), CPU/RAM utilization, beam-search throughput (queries/s for 1/8/16/32 processes), runtime, and generated data size, and send you a summary before starting the full runs.
>
> My SSH public key: `<paste ~/.ssh/id_ed25519.pub here>`
>
> Best regards,
> Kien

(Tạo key nếu chưa có: `ssh-keygen -t ed25519 -C "kienduong160@gmail.com"`, sau đó gửi nội dung file `~/.ssh/id_ed25519.pub`. **Không bao giờ** gửi file không có đuôi `.pub`.)

---

## 5. Thứ tự đề xuất
1. Gửi mail cho chị Linh (mục 4.6) và lấy SSH key. Việc này không phụ thuộc gì khác.
2. Local: thêm sharding và script profiling (mục 4.4), chạy ablation MIND-direct và các chẩn đoán ở mục 3.1. Các việc này chạy được trên Mac, trong vài giờ.
3. Xoá MLP/XGBoost và cập nhật README/report (mục 2).
4. Khi có server: profiling, báo cáo lại cho chị Linh, rồi chạy full data gen và retrieval.
5. Song song: cài đặt baseline A2/A3, sửa A1 (CPU), và chuẩn bị B1/B2/B5 (GPU).
6. Chọn 2–3 cải tiến ở mục 3.2/3.3 dựa trên kết quả chẩn đoán, luôn tuning trên **validation**.

## Nguồn tham khảo
- [A\*Net (NeurIPS 2023)](https://arxiv.org/abs/2206.04798) — [code](https://github.com/DeepGraphLearning/AStarNet)
- [AdaProp (KDD 2023)](https://arxiv.org/abs/2205.15319) — [code](https://github.com/LARS-research/AdaProp)
- [DR.KNOWS (JMIR AI 2025)](https://ai.jmir.org/2025/1/e58670) — [code](https://github.com/serenayj/DRKnows)
- [MedRAG (WWW 2025)](https://arxiv.org/abs/2502.04413)
- NBFNet (Zhu et al., NeurIPS 2021); MINERVA (Das et al., ICLR 2018); MultiHopKG (Lin et al., EMNLP 2018); RED-GNN (Zhang & Yao, WWW 2022); NSM (He et al., WSDM 2021); SR (Zhang et al., ACL 2022); G-Retriever (He et al., 2024); Think-on-Graph (Sun et al., ICLR 2024)
