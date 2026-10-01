# MedMIG-CR — Plan 2

> Ngày: 2026-10-01. Tiếp nối [plan.md](plan.md). Branch hiện tại: `feat/cpu-parallel-retrieval`, chưa commit.

---

## 1. Danh sách baseline chốt cho paper

### Nguyên tắc chung (ghi vào phần Experimental Setup)
- Mọi phương pháp nhận **cùng một tập seed node**, chạy trên **cùng đồ thị PrimeKG** và **cùng tập ứng viên**:
  - **P-closed** (protocol chính): 47 target đã map.
  - **P-open** (phụ, nếu có thời gian): 22,205 disease node.
- Mọi phương pháp đánh giá trên **cùng một danh sách `patient_index`** (lấy mẫu phân tầng theo pathology, seed cố định) và dùng chung evaluator.
- Các baseline có học dùng **cùng một tập train con** (ví dụ 100k query) và tune trên cùng một tập validation con.
- Báo cáo **MRR, R@1/5/10/20/50, candidate coverage, latency/query**, kèm bootstrap CI 95% và paired test so với phương pháp chính.

### 1.1 Bắt buộc
| # | Baseline | Vai trò trong paper | Chạy ở đâu | Ghi chú triển khai |
|---|---|---|---|---|
| 1 | **PPR / RWR** (sửa phần hội tụ) | Lan truyền không tham số kinh điển | **Local CPU, song song** | Code đã có ([run_ppr_baseline.py](Benchmark_baselines/Graph_Retrieval/PPR/run_ppr_baseline.py)). Nên batch nhiều query thành ma trận `[N × B]` để mỗi vòng lặp chỉ là một phép nhân sparse×dense; ~200 iteration, dung sai 1e-6. Ước tính 5000 query mất khoảng 15–30 phút. |
| 2 | **Katz / đếm path cắt ở L hop** | Baseline "graph proximity" thuần cấu trúc | **Local CPU, song song** | Code mới, khoảng 100 dòng. `score = Σ_ℓ β^ℓ · A^ℓ · seed_vec`, dùng chung hạ tầng với PPR. |
| 3 | **Seed-only beam** (đã có) | Ablation và baseline "beam không học" | **Local CPU, song song** | Chạy lại bằng `--num_workers 6` (khoảng 6 phút cho 5000 query). |
| 4 | **R-GCN** (đã có, chạy lại ở P-closed) | GNN quan hệ học biểu diễn | **Local CPU** | Lần trước train trên CPU mất 39 phút (200k query, 10 epoch). Chỉ cần chạy lại với ứng viên là 47 target. |
| 5 | **NBFNet** (NeurIPS 2021) | GNN dựa trên path, lan truyền từ nguồn theo truy vấn. Baseline học **mạnh nhất và chuẩn nhất** cho bài toán "từ node nguồn tới node đích" | **Server GPU** | Dùng bản PyG ([NBFNet-PyG](https://github.com/KiddoZhu/NBFNet-PyG)). Boundary condition là indicator trên **nhiều seed**, query relation là một relation giả "diagnosis". Lan truyền trên toàn bộ 2.7M cạnh nên mỗi batch tốn nhiều VRAM: 24GB là đủ nếu batch nhỏ. |
| 6 | **A\*Net** (NeurIPS 2023) *hoặc* **AdaProp** (KDD 2023) | Lan truyền **có học cách prune**, là "đối thủ trực tiếp" của beam search có hướng dẫn bằng interest | **Server GPU** | Chỉ cần chọn một. Đề xuất **AdaProp**: viết bằng PyTorch thuần, ít phụ thuộc, dễ chuyển sang multi-seed. A\*Net phụ thuộc TorchDrug, cài trên môi trường mới khá khó. |
| 7 | **MINERVA** (ICLR 2018) | RL agent đi path, cùng ý tưởng "duyệt ra path giải thích được" | **Local CPU được nhưng chậm, server tốt hơn** | Rollout chạy trên CPU. Có thể train trên tập con 50–100k query bằng nhiều process. Lúc suy luận dùng cùng beam width và số hop như phương pháp của ta. |

### 1.2 Ablation bắt buộc (không phải baseline nhưng reviewer sẽ hỏi)
| Ablation | Chạy ở đâu |
|---|---|
| **MIND-direct**: xếp hạng 47 target theo `max_k cos(interest_k, target)`, không search | Local, vài giây |
| K=1 / K=2 / K=3, có và không có GRU (phần lớn đã có) | Local CPU, song song |
| Độ nhạy theo hop (4/6/8/10) và beam width (32/64/128/256), chạy trên **validation** | Local CPU, song song (mỗi cấu hình khoảng 5–10 phút với 2000 query) |
| Mỗi cải tiến search ở mục 3, bật/tắt riêng | Local CPU, song song |

### 1.3 Tuỳ chọn (nếu còn thời gian hoặc reviewer yêu cầu)
| Baseline | Chạy ở đâu | Ghi chú |
|---|---|---|
| PCST subgraph retrieval (kiểu G-Retriever) | Local CPU | Dùng `pcst_fast`, không cần học |
| DR.KNOWS (chỉ phần retriever và path ranker) | Server GPU | Phải chuyển từ UMLS sang PrimeKG, tốn công |
| RED-GNN | Server GPU | Tương tự AdaProp, cùng nhóm tác giả |
| Think-on-Graph / beam do LLM dẫn | API LLM, khoảng 500 query | Tốn tiền, chỉ đưa vào discussion |

### 1.4 Tổng hợp nơi chạy
- **Local Mac, CPU song song:** PPR, Katz, Seed-only, R-GCN, mọi ablation và sensitivity, MINERVA (chậm).
- **Cần server GPU:** NBFNet, AdaProp hoặc A\*Net (và tuỳ chọn DR.KNOWS, RED-GNN). Đây chính là lý do để thuê GPU sau này, như đã viết trong [mail.md](mail.md).
- **Lưu ý về môi trường:** `.venv` local đang chạy **Python 3.14**, chưa có `torch_scatter`. Các repo GNN baseline thường cần `torch_scatter` hoặc kernel CUDA riêng, và hiếm khi có wheel cho 3.14. Trên server nên dùng **Python 3.11 + PyTorch 2.x + CUDA 12.x**.

### 1.5 Thứ tự làm
1. MIND-direct, PPR (đã sửa) và Katz: rẻ, chạy local, khoảng 1–2 ngày.
2. R-GCN ở P-closed và chạy lại Seed-only trên tập đánh giá cố định.
3. Viết sẵn NBFNet và AdaProp, chạy thử trên subgraph nhỏ ở local, rồi mới thuê GPU.
4. MINERVA.

---

## 2. Xoá MLP và XGBoost (ĐÃ LÀM)
- `git rm -r Benchmark_baselines/Direct/` (11 file được track), và xoá luôn thư mục trên đĩa, gồm khoảng 527MB output/checkpoint bị gitignore.
- [README.md](README.md):
  - Bỏ hai dòng MLP/XGBoost trong "Baseline snapshot" và bỏ đoạn "Interpretation of direct ranking".
  - Viết lại câu "core comparison" thành: mọi phương pháp đều xuất phát từ seed và đi qua đồ thị.
  - Thêm `--num_workers 6` vào lệnh reproduce, kèm một đoạn giải thích.
  - Thêm `parallel.py` và `profile_command.py` vào Code map.
- [requirements.txt](requirements.txt): bỏ `xgboost`, thêm `psutil`. (Repo không có `requirements.md`, mình hiểu bạn muốn nói `requirements.txt`.)
- **Chưa sửa** [report.md](report.md) mục 1.1–1.2, vì đó là báo cáo tuần mang tính lịch sử. Báo mình nếu muốn xoá luôn.

---

## 3. Hai cải tiến search đáng thử nhất

Bối cảnh: nút thắt là **reachability**. Khoảng 46% query không bao giờ chạm tới target (R@20 = R@50 = 0.544), và 46% train query không có path dương cho GRU. Vì vậy ưu tiên những thay đổi làm **tăng coverage**, sau đó mới tới thứ hạng.

**Bước 0 (bắt buộc, khoảng nửa ngày): chẩn đoán trên 2000 query validation.**
- Khoảng cách BFS từ seed tới target.
- Target bị loại khỏi beam ở hop nào.
- Độ dài, relation và node type của các path đúng.

Kết quả chẩn đoán dùng để xác nhận hai lựa chọn dưới đây.

### Cải tiến 1: Pruning theo khoảng cách tới tập target (heuristic kiểu A\*)
- **Ý tưởng.** Chạy BFS đa nguồn một lần từ 47 target để có `dist(v)` = số hop ngắn nhất từ node `v` tới target gần nhất (82k node, tính trong vài giây). Khi mở rộng beam ở hop `h`:
  - **Bỏ hẳn** những neighbor có `dist(v) > max_hops − h`. Từ những node này chắc chắn không thể tới target, nên việc bỏ đi là chính xác, không làm mất path hợp lệ nào. Slot beam được giải phóng cho path có triển vọng.
  - (Tuỳ chọn) Cộng thêm vào step score một heuristic mềm `−λ · dist(v)`.
- **Vì sao có triển vọng.** Beam 128 trên đồ thị có hub (gene/protein) dễ bị kéo sang vùng không chứa disease. Luật cắt này thẳng tay giải quyết đúng vấn đề reachability.
- **Tính công bằng.** Luật chỉ dùng **tập target cố định** (thông tin mà P-closed đã cho mọi phương pháp), không dùng nhãn của từng query. Phải ghi rõ trong paper. Baseline cũng dùng cùng tập ứng viên này nên vẫn công bằng.
- **Chi phí.** Khoảng 30–50 dòng trong [beam_search.py](src/medmigcr_kg/beam_search.py), cộng một mảng `dist` precompute. Không làm chậm search, có thể còn nhanh hơn vì ít ứng viên hơn.
- **Đánh giá.** Coverage (R@50) và MRR trên validation, thử λ ∈ {0, 0.1, 0.3}.

### Cải tiến 2: Gộp bằng chứng từ nhiều path, nhiều seed và nhiều interest
- **Ý tưởng.** Hiện tại mỗi bệnh chỉ lấy điểm của **path tốt nhất** (`max`) ([run_ddxplus_infonce_gru_rerank.py](scripts/retrieval/run_ddxplus_infonce_gru_rerank.py), `beam_endpoint_scores` và `reranked_endpoint_scores`), còn điểm path lại là **tổng** các bước, nên bị lệch theo độ dài. Cách mới:
  - Chuẩn hoá điểm path theo độ dài (trung bình các bước).
  - Điểm bệnh `= logsumexp(τ · score_path) / τ` trên **mọi** path tới bệnh đó, cộng thưởng theo số seed và số interest khác nhau cùng chạm tới bệnh.
  - Trong beam search, `target_paths` hiện chỉ giữ **1 path tốt nhất** cho mỗi target. Cần đổi thành giữ top-m path.
- **Vì sao có triển vọng.** Nhiều triệu chứng cùng dẫn tới một bệnh là bằng chứng mạnh hơn một path đơn lẻ. Đây cũng là lý do PPR và Katz hoạt động được. Cải tiến này nhắm vào **MRR/R@1** (thứ hạng), bổ sung cho Cải tiến 1 vốn nhắm vào coverage.
- **Chi phí.** Thấp. Phần lớn là post-processing; chỉ phải sửa nhỏ để giữ top-m path cho mỗi target.
- **Đánh giá.** MRR và R@1 trên validation, thử τ ∈ {0.5, 1, 2}, m ∈ {1, 4, 16}. Áp dụng cho cả nhánh có và không có GRU.

### Ngoài phần search (để sau)
GRU reranker hiện **không nhận thông tin của truy vấn**: hai bệnh nhân có cùng path sẽ nhận cùng điểm. Lấy interest vector làm hidden khởi tạo cho GRU là thay đổi có khả năng tăng MRR rõ nhất ở khâu rerank. Việc này nên làm sau hai cải tiến trên.

---

## 4. Hình ảnh cho paper

### 4.1 Danh sách hình đề xuất
| # | Hình | Nội dung | Dữ liệu cần có |
|---|---|---|---|
| F1 | **Teaser / motivation** (đầu paper) | Một bệnh nhân thật trong DDXPlus: triệu chứng → seed trên PrimeKG → 2–3 path giải thích → chẩn đoán đúng, đặt cạnh "black-box classifier chỉ trả về nhãn" | Một case study thật |
| F2 | **Kiến trúc tổng thể** (bắt buộc) | 4 khối: (a) evidence mapping → seed; (b) ClinicalMIND với capsule routing → K interest → projection vào không gian node2vec; (c) semantic beam search hai chiều trên PrimeKG, step score; (d) GRU path reranker → danh sách bệnh. Ghi rõ đâu là frozen, đâu là trainable, loss nào | Không |
| F3 | **Chi tiết MIND + InfoNCE** (có thể gộp vào F2) | Behavior-to-Interest routing, K capsule, InfoNCE trên target và evidence, diversity loss | Không |
| F4 | **Recall@k curve** | Đường R@k (k = 1…50) cho phương pháp chính và mọi baseline. Cho thấy rõ khoảng cách và "trần coverage" | Predictions của tất cả phương pháp |
| F5 | **Phân tích reachability** | (a) Coverage theo số hop; (b) phân bố độ dài path đúng; (c) target bị prune ở hop nào. Hình này dùng để biện minh cho Cải tiến 1 | Bước 0 ở mục 3 |
| F6 | **Ablation / sensitivity** | MRR và R@50 theo K (1/2/3), theo beam width và theo số hop | Các run trên validation |
| F7 | **Hiệu quả vs độ chính xác** | Scatter: trục x là latency/query (log), trục y là MRR; mỗi điểm là một phương pháp | Summary JSON |
| F8 | **Hình ảnh hoá multi-interest** | UMAP các interest vector cùng embedding 47 target, tô màu theo pathology. Thêm ví dụ một bệnh nhân có 3 interest đi về 3 vùng bệnh khác nhau | Interest vectors trên tập test |
| F9 | **Case study path** (định tính) | 2–3 bệnh nhân, top path sau rerank, có tên relation (vd. `phenotype —phenotype_present→ disease`). So sánh một case đúng với một case sai | Predictions và path |
| F10 | **Hiệu năng theo pathology** (phụ lục) | Heatmap hoặc bar chart R@1 cho 49 pathology, so sánh với baseline mạnh nhất | Per-query metrics |
| T | **Bảng** | Thống kê dataset và mapping; bảng kết quả chính; bảng ablation | Đã có phần lớn |

Tối thiểu cho bài báo chính: **F2, F4, F5, F6, F9** và bảng kết quả. Các hình còn lại đưa vào nếu còn chỗ hoặc vào phụ lục.

### 4.2 Công cụ và nguồn
**Sơ đồ kiến trúc (F1, F2, F3):**
- **draw.io / diagrams.net** (miễn phí): xuất **PDF/SVG vector**, đủ dùng cho paper. Đây là lựa chọn phổ biến nhất.
- **Figma**: đẹp hơn, linh hoạt hơn, có bản miễn phí.
- **TikZ** (LaTeX): đồng bộ font tuyệt đối với paper nhưng tốn thời gian. Hợp với hình đơn giản như F3.
- **Excalidraw**: phác thảo nhanh trước khi vẽ bản chính.
- **Icon:** [Bioicons](https://bioicons.com) (CC, chuyên y sinh), Font Awesome, [Health Icons](https://healthicons.org) (CC0). Nếu giấy phép yêu cầu thì phải ghi nguồn. Tránh icon kiểu clip-art.
- **Quy tắc:** luôn xuất vector (PDF/SVG), font khớp với template, bảng màu nhất quán và phân biệt được cho người mù màu (vd. Okabe–Ito). Mọi hình phải đọc được khi in đen trắng.

**Biểu đồ thực nghiệm (F4–F8, F10):**
- **matplotlib** (+ seaborn hoặc gói `SciencePlots`), xuất PDF. Viết script trong `scripts/figures/` để **tái tạo được từ file JSON/CSV kết quả**. Reviewer và chính bạn sẽ cần vẽ lại khi số liệu thay đổi.
- F8 dùng `umap-learn`. F9 dùng `networkx` + Graphviz để vẽ subgraph, rồi chỉnh lại trong draw.io hoặc Inkscape.

**Dùng Claude (artifact hoặc code) có được không?**
- **Được, với vai trò công cụ hỗ trợ vẽ**, theo hai cách:
  1. **Claude viết code vẽ hình** (matplotlib, TikZ, Graphviz, SVG). Đây là cách mình khuyên nhất: kết quả nằm trong repo, tái tạo được, bạn kiểm soát từng con số. Mình có thể viết luôn `scripts/figures/*.py` cho F4–F8.
  2. **Claude artifact** (trang HTML/SVG) để **phác thảo nhanh** bố cục sơ đồ kiến trúc, xem trước, rồi sửa. Sau đó lưu phần SVG và mở bằng draw.io, Figma hoặc Inkscape để chỉnh font, kích thước và xuất PDF. Artifact là trang web nên không dùng trực tiếp làm hình paper; chỉ nên dùng làm bản nháp hoặc bản SVG gốc.
- **Không nên** dùng AI tạo ảnh raster (DALL·E, Midjourney, Gemini image…) cho hình trong paper. Nhiều nhà xuất bản (Springer Nature, Elsevier, IEEE…) cấm hoặc hạn chế ảnh do generative AI tạo ra. Ảnh kiểu này cũng hay sai chi tiết kỹ thuật.
- **Luôn kiểm tra chính sách AI của venue bạn nộp** và ghi chú việc dùng AI hỗ trợ (code, viết, hình) theo yêu cầu của venue, thường là ở phần Acknowledgements hoặc Methods.

### 4.3 Đề xuất thực hiện
1. Phác F2 bằng tay hoặc Excalidraw → mình dựng bản SVG (qua artifact hoặc code) → bạn hoàn thiện trong draw.io.
2. Mình viết `scripts/figures/` cho F4, F6, F7 ngay khi có predictions của các baseline. F5 vẽ sau bước chẩn đoán.
3. F9 chọn case sau khi đã có kết quả cuối.

---

## 5. Việc tiếp theo cần bạn quyết định
- [ ] Commit các thay đổi trên branch `feat/cpu-parallel-retrieval` (chạy song song, profiling, xoá Direct, README). Có thể tách thành 2 commit.
- [ ] Có xoá MLP/XGBoost khỏi `report.md` không?
- [ ] Chọn AdaProp hay A\*Net cho baseline số 6.
- [ ] Bắt đầu với: (a) bước chẩn đoán + MIND-direct, (b) PPR/Katz batched, hay (c) Cải tiến 1.
- [x] Chốt protocol dữ liệu: train/path GRU 50k, valid 5k, test 10k (xem mục 6).

---

## 6. Protocol dữ liệu đã chốt (2026-10-01)

Một protocol dùng chung cho **main model, mọi ablation và mọi baseline**:

| Vai trò | Tập | File | Ghi chú |
|---|---|---|---|
| Train (MIND, GRU, mọi baseline có học) | 50,000 / 1,023,037 train | `data/processed/ddxplus_v2_subsets/train_50k.csv` | TV so với toàn bộ train = 0.0001 |
| Sinh path cho GRU | **Cùng** 50k ở trên | (như trên) | Path được sinh bằng checkpoint MIND của từng cấu hình |
| Validation (early stopping, tune hop/beam/λ/τ) | 5,000 / 132,190 valid | `data/processed/ddxplus_v2_subsets/valid_5k.csv` | **Không tune trên test** |
| Test (mọi con số trong paper) | 10,000 / 134,236 test | `data/processed/ddxplus_v2_subsets/test_10k.csv` | TV = 0.0005; với 10k query, R@k có sai số khoảng ±1 điểm (CI 95%) |

- Ba file được tạo bằng [build_query_subsets.py](scripts/preprocess/build_query_subsets.py): lấy mẫu phân tầng theo pathology, seed 42, đủ 49/49 pathology, giữ nguyên thứ tự dòng gốc. Chạy lại cho ra file giống hệt (đã kiểm tra md5). Danh sách `patient_index` nằm trong `subsets_summary.json`.
- Dùng bằng cách đưa thẳng các file này vào `--train_csv`, `--valid_csv`, `--queries_csv`, `--test_queries_csv`, `--test_csv` của script tương ứng. Bỏ `--limit_patients` và `--max_*_rows`.
- **Hệ quả:**
  - Checkpoint MIND E10 cũ (train full) và file path 100k cũ **không còn là main model**.
  - Checkpoint cũ chuyển thành ablation tham chiếu **"MIND-full"**.
  - File 100k cũ không dùng lại được: tập train_50k được lấy ngẫu nhiên trên toàn bộ 1M query, chỉ khoảng 5k query trùng với 100k dòng đầu.
- **Ghi chú cho paper:** GRU được train trên path sinh từ chính các query MIND đã thấy khi train. Interest trên train có thể "dễ" hơn trên test. Đây là cách làm chuẩn nhưng nên nêu trong phần Limitations.
- Kết quả cũ trên "5000 dòng đầu" (README, `experiments/ablations/`) **phải chạy lại** trên `test_10k.csv` để mọi số trong paper dùng cùng cohort.

### 6.1 Danh sách run
| Nhóm | Run | Train | Path GRU | Đánh giá |
|---|---|---|---|---|
| **Main** | MIND K=3 (50k) + beam + GRU post-hoc | train_50k | sinh từ MIND-50k K=3 | test_10k |
| Ablation | MIND K=3 (50k) + beam, không GRU | train_50k | — | test_10k |
| Ablation | MIND K=1 (50k) ± GRU | train_50k | sinh từ MIND-50k K=1 | test_10k |
| Ablation | MIND K=2 (50k) ± GRU (tuỳ chọn) | train_50k | sinh từ MIND-50k K=2 | test_10k |
| Ablation | **MIND-full** K=3 (checkpoint E10 cũ) ± GRU | 1.02M (cũ) | sinh lại trên train_50k bằng checkpoint cũ | test_10k |
| Ablation | MIND-direct (không search) cho K=1 và K=3 | train_50k | — | test_10k |
| Ablation | Seed-only beam | — | — | test_10k |
| Sensitivity | Hop {4,6,8,10} × beam {32,64,128,256} | — | — | **valid_5k** |
| Cải tiến | Pruning theo khoảng cách; gộp nhiều path | — | sinh lại nếu cần | tune trên valid_5k, báo cáo trên test_10k |
| Baseline | PPR, Katz | — | — | test_10k |
| Baseline | R-GCN, MINERVA | train_50k | — | test_10k |
| Baseline (GPU) | NBFNet, AdaProp/A\*Net | train_50k | — | test_10k |

### 6.2 Ước tính thời gian trên Mac (6 worker)
| Bước | Thời gian |
|---|---|
| Sinh path GRU 50k query, K=3 | ~1 giờ (K=1: ~20 phút) |
| Retrieval test_10k, K=3 | ~13 phút (K=1: ~5 phút) |
| Retrieval valid_5k (mỗi cấu hình sensitivity) | ~5–10 phút |
| Train MIND 50k, train GRU | Chưa đo. MIND cũ train full trên CPU, nên với 50k sẽ nhanh hơn nhiều. Sẽ đo ở lần chạy đầu. |

Toàn bộ main + ablation (không tính baseline GPU) ước tính khoảng **1–2 ngày máy**, chia ra chạy qua đêm.

### 6.3 Thứ tự chạy
1. Train MIND-50k K=3 và K=1 (song song nếu RAM cho phép).
2. Sinh path GRU trên train_50k cho K=3 và K=1, rồi train hai GRU.
3. Chạy retrieval + evaluate trên test_10k: main, các ablation K và seed-only, MIND-direct.
4. MIND-full: sinh path trên train_50k bằng checkpoint E10 cũ, train GRU, đánh giá trên test_10k.
5. PPR/Katz và R-GCN trên cùng protocol.
6. Sensitivity và các cải tiến search trên valid_5k, sau đó áp cấu hình tốt nhất lên test_10k.
7. Baseline GPU khi đã có server.
