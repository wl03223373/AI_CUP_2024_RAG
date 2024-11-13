import os
from rapidocr_pdf import PDFExtracter
from opencc import OpenCC

# 流程
# 1.文字直接讀出 or 掃描檔用ocr轉成文字
# 2.一律轉繁體中文
# 3.存成txt檔

def ocr(file_path):
    """將PDF中文字擷取、同時轉換成繁體中文."""
    pdf_extracter = PDFExtracter()
    texts = pdf_extracter(file_path, force_ocr=False)
    tmp = []
    # 轉繁體中文
    cc = OpenCC('s2tw')

    for text in texts:
    	# 不分頁 全部存成一整段文字
        tmp.append(cc.convert(text[1]))

    return tmp

def process_folder(folder_path):
    """遞迴存取資料集目錄抽取PDF文字，並令存成TXT檔."""
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            texts = ocr(pdf_path)

            # Save the OCR result to a text file
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(folder_path, txt_filename)

            # 存成txt檔
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(texts))
            print(f"Processed and saved: {txt_filename}")

# Folder containing PDF files
folder_path = r"D:\AIcup\競賽資料集\reference\finance"

process_folder(folder_path)
