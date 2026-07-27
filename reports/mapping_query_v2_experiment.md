# Mapping and Query-Building V2 Experiment Report

Ngày chạy: 2026-07-26  
Knowledge graph: PrimeKG  
Clinical benchmark: DDXPlus  
Evaluation cohort: 5.000 test queries; 4.999 patient IDs trùng với cohort 5.000 của run cũ.

## Mục tiêu

Thí nghiệm này xử lý hai bottleneck đã xác định trong workflow cũ:

1. Mapping DDXPlus evidence/pathology sang PrimeKG quá ít và dễ sai do fuzzy matching nguyên câu.
2. Query builder làm mất semantic diversity, giữ rất ít seed node và không kiểm tra target leakage.

Sau khi sửa, toàn bộ workflow được chạy lại từ query generation, MIND K=1/2/3, linear alignment, beam retrieval đến Recall@K và MRR.

## Phương pháp cũ

Mapping cũ chuẩn hóa text rồi lấy điểm lớn nhất của `SequenceMatcher`, token Jaccard và substring containment. Mỗi concept chỉ auto-select top-1 nếu vượt threshold. Evidence question dài được so trực tiếp với tên HPO, categorical value âm không được hiểu về ngữ nghĩa, và pathology chỉ tìm trong disease nodes.

Query builder cũ gộp toàn bộ mapped evidence thành một set seed node. Nó không báo evidence mention coverage, mức collapse hay seed-target overlap. Kết quả là checkpoint chỉ có 41 symptom IDs gồm PAD/UNK, tức 39 PrimeKG seed nodes thực; disease vocabulary có 35 target nodes thực và alignment chỉ có 35 anchors.

Beam cũ dùng width 16, depth 2 và chỉ rank disease endpoint còn nằm trong top returned paths. Vì vậy Recall@20 và Recall@50 bị giới hạn bởi candidate generation, thường bằng Recall@5.

## Phương pháp mới

Mapping v2 trong `scripts/mapping/build_ddxplus_primekg_mappings.py` bổ sung:

- Exact clinical aliases cho abbreviation/paraphrase như GERD, PSVT, SLE, HIV, NSTEMI/STEMI và COPD exacerbation.
- Clinical phrase aliases cho evidence như shortness of breath -> Dyspnea và runny nose -> Rhinorrhea.
- Token inverted index để fuzzy matching chỉ chấm các PrimeKG candidates có token liên quan.
- Negative-value filter cho categorical evidence như `N`, `No`, `False`, `0`.
- Tối đa hai evidence nodes nếu cùng vượt threshold và nằm trong margin 0,03 so với top-1.
- Condition target có thể là MONDO disease hoặc HPO clinical condition khi PrimeKG không biểu diễn nó dưới type disease.

Query builder v2 trong `scripts/preprocess/build_ddxplus_test_queries.py` bổ sung coverage/diversity statistics và loại mọi target node khỏi seed nodes. Audit cuối phát hiện và loại 19.868 train, 2.208 validation và 2.222 test target-seed overlaps.

Beam search v2 giữ width cho traversal nhưng thu thập mọi target node gặp trong neighbor expansions. Điều này tách hai khái niệm: beam width quyết định path nào tiếp tục được mở rộng, còn target-aware collection quyết định candidate nào được đưa vào ranking. Final setup dùng depth 3, width 64, tối đa 256 normal paths mỗi interest và target universe 47 nodes.

## Data Coverage

| Chỉ số | Cũ | V2 sạch |
|---|---:|---:|
| Auto-mapped evidence entries | 648 / 987 | 725 / 987 |
| Negative categorical entries filtered | 0 | 9 |
| Unique evidence nodes trong mapping | 39 được MIND sử dụng | 168 trong mapping |
| Train patients giữ lại | 786.229 / 1.025.602 | 1.023.037 / 1.025.602 |
| Test patients giữ lại | 103.575 / 134.529 | 134.236 / 134.529 |
| Patients mất vì pathology mapping, test | 30.600 | 0 |
| Unique seed nodes trong train queries | 39 | 132 |
| Unique target nodes | 35 | 47 |
| Median seed nodes/query | Không được log | 9 |
| Evidence mention coverage, test | Không được log | 74,16% |
| Alignment anchors | 35 | 47 |

V2 map đủ 49 pathology labels nhưng chỉ có 47 PrimeKG targets vì hai cặp collapse theo ontology:

- Allergic sinusitis và Acute rhinosinusitis -> `disease|MONDO|5961`.
- Stable angina và Unstable angina -> `effect/phenotype|HPO|1681`.

## Training and Alignment

Ba model được train từ đầu với D=64, routing iterations=3, 3 epochs, batch size 256 và 8 random negatives. Dataset và negative sampling được vector hóa để tránh parse lại CSV ở từng epoch; loss objective không thay đổi.

| K | Final train loss | Final validation loss | Mean inter-interest cosine | Alignment cosine after |
|---:|---:|---:|---:|---:|
| 1 | 0,0148 | 0,0152 | N/A | 0,9790 |
| 2 | 0,0143 | 0,0148 | 0,9398 | 0,9821 |
| 3 | 0,0139 | 0,0126 | 0,9392 | 0,9746 |

K=2 và K=3 có cosine khoảng 0,94, cho thấy interests gần collapse về cùng hướng. Validation loss tốt không đủ chứng minh multi-interest representation đã tách biệt.

## Final Retrieval Results

Metrics dưới đây dùng evaluator đã sửa: query không có prediction vẫn được tính zero thay vì bị loại khỏi denominator.

| Method | MRR | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| Old K=1 | 0,0107 | 0,0112 | 0,0112 | 0,0112 | 0,0112 |
| Old K=2 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| Old K=3 | 0,0086 | 0,0112 | 0,0112 | 0,0112 | 0,0112 |
| V2 K=1 | 0,0490 | 0,1008 | 0,1054 | 0,1054 | 0,1054 |
| V2 K=2 | **0,0905** | **0,1274** | **0,1398** | **0,1398** | **0,1398** |
| V2 K=3 | 0,0565 | 0,1036 | 0,1082 | 0,1082 | 0,1082 |

K=2 là model tốt nhất trong run này. So với K=1 V2, K=2 tăng MRR khoảng 84,7% và Recall@10 khoảng 32,6%. K=3 chỉ tăng Recall@10 khoảng 2,7% so với K=1 và kém K=2 rõ rệt.

Candidate generation vẫn là bottleneck. Số target candidates trung vị chỉ là 5/query; maximum là 12. Vì thế Recall@20 và Recall@50 vẫn bằng Recall@10. Đây là reachability limitation của beam/PrimeKG subgraph, không phải bằng chứng rằng ranking đã bão hòa ở K=10.

| K | Queries có prediction | Mean candidates | Median candidates | Max candidates |
|---:|---:|---:|---:|---:|
| 1 | 4.987 / 5.000 | 4,94 | 5 | 12 |
| 2 | 4.984 / 5.000 | 4,41 | 5 | 12 |
| 3 | 5.000 / 5.000 | 5,15 | 5 | 12 |

## Kết luận

Mapping/query v2 đã giải quyết bottleneck coverage: pathology loss giảm từ 30.600 test cases xuống 0, train coverage tăng từ 76,7% lên 99,75%, seed vocabulary tăng hơn ba lần và anchors tăng từ 35 lên 47. End-to-end retrieval tăng mạnh so với run lịch sử.

Tuy nhiên, không thể gán toàn bộ mức tăng cho mapping vì final pipeline đồng thời sửa evaluator và candidate extraction. Kết luận an toàn là workflow cũ bị giới hạn nghiêm trọng ở data interface và beam output; sau khi sửa hai lớp này, MIND mới bắt đầu tạo signal đo được.

K=2 hiện là lựa chọn baseline hợp lý nhất. K=3 chưa chứng minh được lợi ích do capsule collapse và chi phí retrieval tăng gần tuyến tính theo K.

## Rủi ro và bước tiếp theo

- Clinical aliases và fuzzy selections cần được chuyên gia hoặc ontology crosswalk xác nhận; hiện chưa có gold mapping annotation.
- Hai pathology pairs bị collapse cần quyết định: chấp nhận parent concept, hỗ trợ multi-target hay loại khỏi benchmark chính.
- Cần diversity regularization hoặc multi-positive/self-supervised objective để giảm inter-interest cosine từ khoảng 0,94.
- Cần target-aware beam rộng/depth-stratified experiment hoặc candidate completion để Recall@20/@50 có candidate support thực.
- Cần rerun full 134.236-query test sau khi chốt hyperparameters; kết quả hiện tại là cohort 5.000 để kiểm chứng workflow.
- Cần ablation riêng: mapping v2 + beam cũ, mapping cũ + target-aware beam, và mapping v2 + target-aware beam để định lượng causal contribution.

## Artifacts

- Mapping v2: `data/mappings/ddxplus_v2/`
- Processed train/valid/test: `data/processed/ddxplus_v2/`
- Paired cohort: `data/processed/comparison_cohort/`
- Checkpoints/projections: `artifacts/checkpoints/ddxplus_v2/`
- Final K=1: `results/mind_v2_targetaware_k1_eval5000/`
- Final K=2: `results/mind_v2_targetaware_k2_eval5000/`
- Final K=3: `results/mind_v2_targetaware_k3_eval5000/`

