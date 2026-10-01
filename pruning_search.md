# Target-Distance Guided Beam Search (pruning + heuristic kiểu A\*)

> Ngày: 2026-10-01 · Branch `feat/cpu-parallel-retrieval` · Code: [target_distance.py](src/medmigcr_kg/target_distance.py), [beam_search.py](src/medmigcr_kg/beam_search.py)

## TL;DR
- **Không phải data leakage**, với điều kiện ghi rõ trong paper: phương pháp chỉ dùng **tập nhãn cố định** (47 target, giống nhau cho mọi bệnh nhân) và cấu trúc đồ thị. Nó không dùng nhãn của từng bệnh nhân.
- **Trên 1000 query validation (không có GRU), MRR tăng từ 0.404 lên 0.749 và coverage từ 0.584 lên 0.977** với cấu hình `--distance_pruning --distance_weight 0.3`.
- **Không dùng được với GRU hiện tại:** MRR giảm từ 0.465 xuống 0.307. GRU phải được train lại trên path sinh bằng A\*, và nên được thêm thông tin truy vấn.
- **Cảnh báo lớn hơn:** MIND-direct (xếp 47 target theo cosine với interest, không search) đạt **MRR 0.990** trên cùng 1000 query. Mọi biến thể search hiện tại đều thấp hơn. Xem mục 7.
- **Về novelty:** không nên coi đây là đóng góp chính. Nên mô tả như một thành phần của thuật toán search, kèm ablation. Xem mục 8.

---

## 1. Cách chạy

Hai tham số mới (mặc định tắt, nên hành vi cũ giữ nguyên) có trong mọi script dùng beam search:

| Tham số | Ý nghĩa |
|---|---|
| `--distance_pruning` | Bỏ các neighbor không thể chạm bất kỳ target nào trong số hop còn lại |
| `--distance_weight λ` | Heuristic A\*: ưu tiên trong beam là `g − λ·d_T(node hiện tại)`. Đề xuất **λ = 0.3** |

Cả hai đều cần `--condition_map` (để biết tập target). Mảng khoảng cách được tính khi khởi động, mất khoảng 2 giây cho mỗi process.

**Retrieval không có GRU** (cấu hình tốt nhất hiện tại):
```bash
.venv/bin/python scripts/retrieval/run_ddxplus_infonce_retrieval.py \
  --test_queries_csv data/processed/ddxplus_v2_subsets/valid_5k.csv \
  --checkpoint <mind_checkpoint.pt> --condition_map data/mappings/ddxplus_v2/condition_to_primekg.json \
  --output_csv <out>/predictions.csv \
  --interest_count 3 --max_hops 10 --beam_width 128 --paths_per_interest 4096 --top_k 50 --alpha 1.0 --beta 0.1 \
  --distance_pruning --distance_weight 0.3 \
  --device cpu --graph_device cpu --num_workers 6
```
Sau đó đánh giá bằng `scripts/evaluation/evaluate_ddxplus_retrieval.py` như bình thường.

**Các script khác có cùng hai tham số:**
- `scripts/retrieval/run_ddxplus_infonce_gru_rerank.py`
- `scripts/reranking/build_gru_path_reranker_dataset.py`: sinh path cho GRU bằng A\*. Bắt buộc dùng nếu muốn GRU khớp với search mới.
- `Benchmark_baselines/Graph_Retrieval/SeedOnlyBeam/run_seed_only_beam.py`: để ablation seed-only dùng cùng thuật toán search với main.

Các tham số này được ghi vào file summary JSON (`distance_pruning`, `distance_weight`).

**Lưu ý về thời gian:** với λ = 0.3, mỗi query chậm hơn khoảng 3.5 lần (1000 query mất 274s so với 77s, 6 worker). Lý do là beam đi sâu hơn vào vùng có nhiều path tới target. Ước tính test_10k mất khoảng 45 phút với 6 worker.

---

## 2. Động cơ

Phân tích trước đó cho thấy **nút thắt là reachability, không phải khâu xếp hạng**:
- Pipeline E10 K=3: R@20 = R@50 = 0.544 trên test. Khoảng 46% query không bao giờ chạm tới target.
- Data cho GRU: 46% query train không có path dương nào.

Beam search chọn neighbor theo `α·cos(e_v, z) − β·log(deg(v)+1)`. Điểm này chỉ đo **mức tương đồng ngữ nghĩa với interest**, không biết node đó có dẫn tới bệnh nào hay không. Trên PrimeKG, beam rộng 128 dễ bị kéo vào các vùng gene/protein, pathway, drug: những node có embedding gần interest nhưng **xa tập bệnh đích**. Khi đó slot beam bị chiếm bởi các path không thể kết thúc ở một chẩn đoán.

Phân bố khoảng cách hop từ mọi node tới target gần nhất (đo trên PrimeKG):

| d_T | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | ≥11 | ∞ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Số node | 47 | 2,246 | 31,386 | 27,460 | 8,702 | 4,607 | 3,075 | 2,425 | 1,246 | 583 | 350 | 100 | 13 |

Seed của bệnh nhân thường rất gần tập target: trên 1000 query, seed gần nhất cách target 0–2 hop. Tuy vậy, khi beam đi xa ra (gene, pathway…) thì có thể cách target 4–10 hop. Đó chính là những path vô ích mà phương pháp này loại bỏ hoặc hạ ưu tiên.

---

## 3. Có phải data leakage không?

**Định nghĩa:** leakage là khi mô hình dùng thông tin mà lúc suy luận thật sự không có, chủ yếu là **nhãn của chính query đang dự đoán** hoặc dữ liệu test.

| Thông tin phương pháp dùng | Có phụ thuộc nhãn của query không? | Lúc suy luận có sẵn không? |
|---|---|---|
| Đồ thị PrimeKG (cấu trúc) | Không | Có |
| Tập target T = 47 node, hợp của mọi pathology trong `condition_to_primekg.json` | **Không**: giống hệt nhau cho mọi bệnh nhân. Được xây từ metadata của DDXPlus (`release_conditions.json`), không từ bệnh nhân nào | Có: đó là **không gian nhãn** của bài toán |
| Nhãn đúng của từng bệnh nhân (`target_node_keys`) | — | **Không được dùng**: chỉ evaluator đọc |

**Kết luận: không phải leakage.** Đây là dùng **tiên nghiệm về không gian nhãn (label-space prior)**. Thông tin này protocol closed vốn đã cho mọi phương pháp:
- Mọi baseline (PPR, Katz, R-GCN) chỉ xếp hạng trong 47 target.
- MIND được train InfoNCE trên chính 47 target.
- Pipeline cũ đã dùng T để thu path (`target_node_ids`) và lọc endpoint.

Phương pháp mới chỉ dùng T **sớm hơn**: trong lúc search, chứ không chỉ ở bước lọc cuối.

**Những điều bắt buộc ghi trong paper để reviewer không bắt lỗi:**
1. Nói rõ search được dẫn hướng bởi tập nhãn cố định T, và T không phụ thuộc vào từng query.
2. Mọi baseline được đánh giá trên cùng không gian ứng viên T (đã đúng).
3. Phương pháp **không áp dụng nguyên dạng cho protocol open**: khi đó T = 22,205 bệnh, d_T của hầu hết node rất nhỏ nên tác dụng cắt gần như mất. Đây là giới hạn cần nêu.
4. **Không** tune λ trên test. λ = 0.3 được chọn trên validation (1000 query đầu của `valid_5k.csv`).

**Lưu ý phân biệt:** nếu dùng khoảng cách tới **target của chính bệnh nhân** thì đó là leakage rõ ràng. Code chỉ nhận `target_universe`, là hợp của mọi pathology, và không có đường nào truyền nhãn của query vào search.

---

## 4. Toán học

### 4.1 Ký hiệu
- Đồ thị vô hướng G = (V, E), là hợp của các cạnh PrimeKG với chiều ngược của chúng. Đây đúng là đồ thị mà beam duyệt, vì beam mở rộng cả neighbor vào và ra.
- Tập target cố định T ⊂ V, với |T| = 47.
- Với query q: tập seed S_q ⊂ V và K interest z_1, …, z_K ∈ ℝ⁶⁴ do MIND sinh ra.
- Độ sâu tối đa H (`max_hops`), độ rộng beam B (`beam_width`).
- Path p = (v_0, v_1, …, v_ℓ) với v_0 ∈ S_q, độ dài ℓ, đỉnh cuối end(p) = v_ℓ.

### 4.2 Khoảng cách tới tập target
d_T(v) = min_{t ∈ T} dist_G(v, t), với dist_G là số hop ngắn nhất. Quy ước d_T(v) = ∞ nếu v không liên thông với T.

d_T được tính **một lần cho mọi node** bằng BFS đa nguồn xuất phát từ T, độ phức tạp O(|V| + |E|), với 82k node và khoảng 1.8M cạnh vô hướng mất khoảng 2 giây. Nó **không phụ thuộc query** nên chỉ tính một lần cho cả tập dữ liệu.

### 4.3 Điểm ngữ nghĩa (giữ nguyên như pipeline gốc)
Mỗi bước mở rộng tới v với interest z có điểm
  s(v; z) = α · cos(e_v, z) − β · log(deg(v) + 1),
và điểm của path là tổng các bước: g(p; z) = Σ_{i=1..ℓ} s(v_i; z).

### 4.4 Luật cắt chính xác (exact pruning)
Ở hop h (h = 1, …, H), path đang dài h nên còn r_h = H − h bước. Một neighbor v được giữ lại **khi và chỉ khi** d_T(v) ≤ r_h.

**Mệnh đề (không mất path hợp lệ).** Gọi P*_H là tập mọi path (bắt đầu từ seed, không lặp đỉnh, dài tối đa H) có thể kéo dài để chạm một target t ∈ T trong vòng H hop. Luật cắt không loại bỏ tiền tố nào của bất kỳ path nào trong P*_H.

*Chứng minh.* Xét p ∈ P*_H kết thúc ở t ∈ T tại độ dài L ≤ H, và tiền tố của nó tới v_h (h ≤ L). Đoạn path từ v_h tới t có L − h ≤ H − h cạnh, nên dist_G(v_h, t) ≤ H − h. Suy ra d_T(v_h) ≤ dist_G(v_h, t) ≤ r_h, nên v_h không bị cắt. ∎

Vì beam có độ rộng cố định, việc cắt giải phóng slot cho những path còn có thể chạm target. Đây là nguồn tăng coverage (0.584 lên 0.866 chỉ với cắt cứng).

*Ghi chú:* mệnh đề nói về **tập path khả thi**. Vì beam là tìm kiếm xấp xỉ, kết quả cuối vẫn có thể khác, nhưng chỉ theo hướng **dành chỗ trong beam cho path khả thi**.

### 4.5 Heuristic kiểu A\* (soft guidance)
Thứ tự trong beam dùng **độ ưu tiên**
  f(p; z) = g(p; z) − λ · d_T(end(p)),
với d_T bị chặn ở H + 1 khi bằng ∞. Trong khi đó, **xếp hạng ứng viên cuối cùng vẫn dùng g**: với endpoint t ∈ T thì d_T(t) = 0 nên f = g.

Tương ứng với A\*:
- g là "chi phí đã đi" (ở đây là điểm cần tối đa hoá).
- λ·d_T là ước lượng "còn bao xa tới đích". d_T là heuristic **admissible** cho số hop còn lại, vì nó không bao giờ đánh giá quá cao số hop thật.
- Khác với A\* kinh điển: beam search có độ rộng giới hạn, và mục tiêu là tối đa hoá điểm ngữ nghĩa thay vì tối thiểu hoá chi phí, nên không có đảm bảo tối ưu. Heuristic ở đây đóng vai trò **định hướng ưu tiên**.

**Vì sao không cộng dồn phạt vào điểm path.** Phiên bản đầu tiên cộng −λ·d_T(v_i) vào **mỗi bước** của g, tức phạt theo lịch sử của path, và vì thế làm sai lệch điểm xếp hạng cuối. Kết quả là R@1 tụt mạnh khi λ lớn: λ = 0.5 cho R@1 0.174. Tách riêng **ưu tiên khi tìm kiếm (f)** và **điểm xếp hạng (g)** sửa được vấn đề này: với λ = 0.5, R@1 lên 0.653.

### 4.6 Các trường hợp đặc biệt
- Khi `distance_pruning` tắt và λ = 0: f = g, mọi neighbor được giữ, thuật toán trùng hoàn toàn với beam gốc.
- Khi chỉ bật cắt cứng (λ = 0): thứ tự beam giữ nguyên theo g, chỉ loại các node không khả thi.

---

## 5. Thuật toán

```
Input : seeds S_q, interests z_1..z_K, graph G, target set T, distance array d_T,
        max hops H, beam width B, weights α, β, λ, pruning flag
Output: best path score per candidate in T

for each interest z_k:
    beam ← {(v) : v ∈ S_q}               g = 0 for every seed path
    found ← ∅                             best path reaching each target
    for h = 1..H:
        r ← H − h
        C ← ∅
        for each path p in beam:
            N ← neighbours_in ∪ neighbours_out of end(p)
            if pruning: N ← {v ∈ N : d_T(v) ≤ r}
            for v in N, v ∉ p:
                g' ← g(p) + α·cos(e_v, z_k) − β·log(deg(v)+1)
                f' ← g' − λ·min(d_T(v), H+1)
                C ← C ∪ {(p ⊕ v, g', f')}
        for each c ∈ C with end(c) ∈ T:  found[end(c)] ← argmax_f(found[end(c)], c)
        beam ← top-B of C by f'
candidate score(t) ← max over interests and paths in found/beam ending at t of g
rank candidates in T by score
```

**Chi phí:**
- Tiền xử lý: một lần BFS đa nguồn, O(|V| + |E|).
- Mỗi bước mở rộng: thêm một phép tra mảng d_T cho mỗi neighbor, chi phí không đáng kể.
- Thời gian thực tế tăng vì beam giữ nhiều path "đi được tới target" hơn, nên số path chạm target cần ghi nhận cũng tăng.

**Cài đặt:**
- `TargetDistanceGuide` trong [target_distance.py](src/medmigcr_kg/target_distance.py) chứa d_T, cờ cắt và λ.
- [beam_search.py](src/medmigcr_kg/beam_search.py) lọc neighbor bằng `admissible` và tính ưu tiên bằng `penalty`.
- `BeamItem.score` = f, `BeamItem.additive_score` = g.
- Với target thì f = g, nên mọi script xếp hạng không phải sửa.

---

## 6. Kết quả trên 1000 query validation

Thiết lập: 1000 query đầu của `valid_5k.csv`, checkpoint MIND E10 K=3 (train trên toàn bộ tập train), H = 10, B = 128, α = 1, β = 0.1, 6 worker, evaluator chung.

### 6.1 Không có GRU (đo riêng tác dụng lên search)
| Cấu hình | MRR | R@1 | R@5 | R@10 | Coverage (R@50) | Thời gian |
|---|---|---|---|---|---|---|
| Beam gốc | 0.404 | 0.351 | 0.484 | 0.515 | 0.584 | 77s |
| Chỉ cắt cứng | 0.463 | 0.385 | 0.558 | 0.629 | 0.866 | 79s |
| Cắt + A\* λ = 0.1 | 0.622 | 0.565 | 0.684 | 0.721 | 0.916 | 110s |
| **Cắt + A\* λ = 0.3** | **0.749** | **0.671** | **0.848** | **0.880** | **0.977** | 274s |
| Cắt + A\* λ = 0.5 | 0.711 | 0.653 | 0.766 | 0.829 | 0.968 | 344s |
| Cắt + A\* λ = 1.0 | 0.693 | 0.637 | 0.744 | 0.802 | 0.945 | 410s |
| *(bản đầu, phạt cộng dồn) λ = 0.1* | 0.587 | 0.518 | 0.648 | 0.712 | 0.964 | 167s |
| *(bản đầu, phạt cộng dồn) λ = 0.3* | 0.506 | 0.375 | 0.719 | 0.780 | 0.942 | 221s |

### 6.2 Có GRU post-hoc (GRU cũ, train trên path của beam gốc)
| Cấu hình | MRR | R@1 | R@10 | Coverage |
|---|---|---|---|---|
| Beam gốc + GRU | 0.465 | 0.426 | 0.536 | 0.584 |
| Cắt + A\* λ = 0.3 + GRU cũ | 0.307 | 0.177 | 0.585 | 0.977 |

GRU cũ làm hỏng kết quả của A\*, vì hai lý do:
1. **Lệch phân bố:** GRU chỉ thấy path của beam gốc khi train.
2. **GRU không nhận thông tin của truy vấn:** token chỉ gồm relation, direction, node type và node. Khi beam đã chạm tới gần hết 47 ứng viên, việc chọn bệnh đúng phải dựa vào bệnh nhân. Một bộ chấm điểm chỉ nhìn hình dạng path không làm được điều đó.

Nếu giữ GRU: sinh lại dữ liệu bằng A\* (`build_gru_path_reranker_dataset.py --distance_pruning --distance_weight 0.3`) **và** thêm interest vector vào GRU.

### 6.3 Mốc tham chiếu: MIND-direct
| Cấu hình | MRR | R@1 | R@10 |
|---|---|---|---|
| MIND-direct (xếp T theo max_k cos(z_k, e_t), không search) | **0.990** | **0.981** | 1.000 |

---

## 7. Hệ quả của MIND-direct (đọc kỹ trước khi viết paper)

MIND được train bằng InfoNCE trực tiếp trên 47 target. Vì vậy trong protocol closed, **nó đã là một bộ phân loại gần như hoàn hảo** (MRR 0.99, khớp với con số 0.991 của MLP closed-set trước đây). Mọi biến thể beam search, kể cả A\* tốt nhất (0.749), đều **làm mất thông tin** so với việc đọc trực tiếp cosine.

Câu hỏi reviewer chắc chắn sẽ đặt: *"Tại sao phải search trên đồ thị nếu chỉ riêng encoder đã cho 0.99?"* Các hướng trả lời, cần bạn chọn:
1. **Đổi giá trị cốt lõi sang tính giải thích được:** MIND (hoặc MIND-direct) xếp hạng chẩn đoán, còn search có dẫn hướng tạo **path giải thích** cho từng chẩn đoán. A\* giúp tìm path tới đúng bệnh đã chọn: coverage 0.977. Khi đó metric chính gồm độ chính xác (giữ ở mức 0.99) và chất lượng hoặc độ phủ của path giải thích.
2. **Kết hợp điểm:** điểm cuối = cos(z, e_t) + γ·(bằng chứng từ path). Phải chứng minh path bổ sung được thông tin, ví dụ ở các ca khó hoặc khi K > 1.
3. **Chuyển sang bối cảnh mà encoder không giải được một mình:**
   - Protocol open (22k bệnh).
   - Bệnh chưa thấy khi train (zero-shot hoặc held-out pathology).
   - Ít nhãn.

   Ở đó cấu trúc đồ thị mới thực sự cần.
4. Không làm gì: paper sẽ rất dễ bị từ chối, vì ablation MIND-direct (bắt buộc phải có) cho thấy search làm kết quả tệ đi.

Mình đề xuất **hướng 1, cộng thêm thí nghiệm của hướng 3 (held-out pathology)**. Đây là quyết định về khung của paper, cần bạn và thầy/chị hướng dẫn thống nhất.

---

## 8. Có nên coi đây là novelty?

**Không nên là đóng góp chính.** Lý do:
- Cắt bằng khoảng cách tới tập đích và dùng heuristic admissible là kỹ thuật **kinh điển** (A\*, goal-directed search, bidirectional search). Trong suy luận trên KG đã có A\*Net (NeurIPS 2023) với hàm ưu tiên **học được**, mạnh hơn heuristic BFS cố định.
- Phương pháp phụ thuộc vào protocol closed (biết trước T nhỏ). Ở protocol open, tác dụng gần như mất. Reviewer sẽ coi đây là chi tiết kỹ thuật.
- Nó không giải quyết khoảng cách với MIND-direct (mục 7).

**Cách trình bày đề xuất trong paper:**
- Mục Method, một đoạn nhỏ: **"Target-aware admissible pruning and A\*-style prioritization"**, kèm công thức d_T, luật cắt, mệnh đề "không mất path khả thi" (2–3 dòng chứng minh), và f = g − λ·d_T.
- Mục Experiments: một bảng ablation (beam gốc / chỉ cắt / cắt + A\*) và một hình coverage theo λ. Đây là bằng chứng rằng **reachability là nút thắt** và được xử lý có nguyên tắc.
- Phần Limitations: phụ thuộc vào tập nhãn cố định, và không hiệu quả trong protocol open.

Nếu paper chuyển sang hướng 1 ở mục 7 (path giải thích), thành phần này trở nên **quan trọng hơn** (tìm path tới đúng chẩn đoán với coverage khoảng 98%). Dù vậy, nó vẫn chỉ là thành phần hỗ trợ, không phải đóng góp chính.

---

## 9. Việc tiếp theo
1. Chạy xác nhận trên toàn bộ `valid_5k` (λ ∈ {0.2, 0.3, 0.4}) để chốt λ. Mỗi cấu hình khoảng 25 phút.
2. Quyết định khung paper theo mục 7.
3. Nếu giữ GRU: sinh lại data bằng A\* và thêm thông tin truy vấn vào GRU.
4. Seed-only ablation cũng nên chạy với `--distance_pruning --distance_weight 0.3` để so sánh có kiểm soát với main.
