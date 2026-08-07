# Medical Safety

> Kết quả chỉ có mục đích nghiên cứu và tham khảo, không thay thế kết luận của bác sĩ hoặc chuyên gia y tế.
>
> For research use only. Not a medical diagnosis.

## Hard rules

1. Not a medical device; not for clinical decision-making.
2. Never store patient identifiers (DICOM PHI stripped on read).
3. Abstain when confidence < threshold.
4. Do not generate unsupported clinical claims in open answers.
5. Attention maps / Grad-CAM-like overlays are **not** definitive explanations.
6. Do not commit raw clinical images or identifiable metadata to git.

## Abstention message

When confidence is below threshold the system returns:

`Không đủ độ tin cậy để đưa ra câu trả lời. Vui lòng kiểm tra lại ảnh hoặc tham khảo ý kiến chuyên gia y tế.`
