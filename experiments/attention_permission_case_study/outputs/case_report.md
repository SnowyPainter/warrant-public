# Attention-Permission Case Study

- Dataset: `hotpotqa`
- Model: `FiD` with Warrant
- Example row: `7122`
- Rank of first supporting passage: `1`
- Support attention ratio: `0.3959`
- Support warranted-mass ratio: `0.4586`
- Matrix: `/workspace/warrant/experiments/attention_permission_case_study/outputs/attention_permission_case.npz`
- Figure PNG: `/workspace/warrant/experiments/attention_permission_case_study/outputs/attention_permission_hotpotqa_case.png`
- Figure PDF: `/workspace/warrant/experiments/attention_permission_case_study/outputs/attention_permission_hotpotqa_case.pdf`

## Question

In what year was the actress who is starred as Baron Frankenstein's new creation in "Frankenstein Created Woman" born?

## Passage-Level Values

| Passage | Support | Attention | Permission | Effective mass | Logit | Title |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| P1 | True | 0.1976 | 0.9622 | 0.1901 | 0.7334 | Frankenstein Created Woman |
| P2 | False | 0.2139 | 0.7951 | 0.1701 | -0.8403 | Earl of Essex |
| P3 | True | 0.1983 | 0.9629 | 0.1909 | 0.8537 | Susan Denberg |
| P4 | False | 0.1643 | 0.7123 | 0.1170 | -0.4405 | Hugh de Courtenay, 4th/12th Earl of Devon |
| P5 | False | 0.2259 | 0.7201 | 0.1626 | -1.3311 | Hugh de Courtenay, 2nd/10th Earl of Devon |

## Interpretation

별표가 붙은 passage는 HotpotQA supporting fact title과 일치하는 supporting evidence passage다. 그림은 raw attention, Warrant permission, 그리고 두 값을 곱한 effective mass를 같은 실제 샘플에서 비교한다. 따라서 이 case study는 Warrant가 attention relevance를 단순히 다시 그리는 것이 아니라, support passage ranking score로 들어가는 weighted value term의 permission을 별도로 조절한다는 정성적 증거로 사용된다.
