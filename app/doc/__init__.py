"""文档识别管线（PDF 扫描页整页 OCR、图片 OCR）。

assemble / labels 是纯逻辑，只依赖标准库和 numpy；engine 依赖 PaddleOCR，只在文档 worker
子进程里导入（requirements-doc.txt，见 notes/doc-ocr-probe.md）。
"""
