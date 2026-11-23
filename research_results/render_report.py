"""Render the Vietnamese draft PDF from verified local result tables and figures.

Requires reportlab, Pillow, pandas. No modeling or changes to input sources.
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle

HERE = Path(__file__).resolve().parent
TABLES, FIGURES = HERE / "tables", HERE / "figures"
OUTPUT = HERE / "output" / "pdf" / "cleanser_results_draft.pdf"
W, H = landscape(A4)
M = 34
CW = W - 2 * M
NAVY = colors.HexColor("#17314b")
TEAL = colors.HexColor("#247f72")
GRAY = colors.HexColor("#536575")
PALE = colors.HexColor("#edf3f6")
GOLD = colors.HexColor("#c6924e")
ASPECTS = ["CleansingEffectiveness", "HydrationSkinComfort", "SkinCompatibilitySafety", "TextureFoaming",
           "Fragrance", "IngredientsFormulation", "Packaging", "PriceValue", "ShopService", "Delivery", "General"]
NAMES = ["Làm sạch / kiểm soát dầu", "Ẩm / cảm giác sau rửa", "Tương thích / an toàn", "Kết cấu / tạo bọt",
         "Mùi hương", "Thành phần / công thức", "Bao bì / đóng gói", "Giá / giá trị", "Dịch vụ shop", "Giao hàng", "Đánh giá chung"]
LABELS = dict(zip(ASPECTS, NAMES))
QSHORT = {"Concentrate Here": "Ưu tiên", "Keep Up the Good Work": "Duy trì", "Low Priority": "Theo dõi", "Possible Overkill": "Cân nhắc"}
MODEL_NAMES = {"bilstm": "BiLSTM", "multitask": "MultiTask", "bigru": "BiGRU", "birnn": "BiRNN"}
SCENARIOS = {"baseline": "CE", "class_weights": "CE trọng số", "focal_loss": "Focal Loss"}


def data(name):
    # None is an explicit internal ABSA state, not a missing table cell.
    return pd.read_csv(TABLES / f"{name}.csv", encoding="utf-8-sig", keep_default_na=False)


def fmt(value, digits=2):
    if isinstance(value, str): return value
    if pd.isna(value): return "NA"
    if isinstance(value, (int,)) or (digits == 0): return f"{float(value):,.0f}"
    return f"{float(value):,.{digits}f}"


class Report:
    def __init__(self):
        for name, file in [("Body", "arial.ttf"), ("BodyBold", "arialbd.ttf"), ("BodyItalic", "ariali.ttf")]:
            pdfmetrics.registerFont(TTFont(name, str(Path("C:/Windows/Fonts") / file)))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="BodyItalic", boldItalic="BodyBold")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        self.c = canvas.Canvas(str(OUTPUT), pagesize=(W, H), pageCompression=1)
        self.c.setTitle("Kết quả nghiên cứu ABSA sữa rửa mặt - Bản nháp")
        self.c.setAuthor("ABSA-T7")
        self.pages = []
        self.p = 0
        self.drawn = []

    def paragraph(self, text, x, top, width, size=10, leading=None, color=NAVY, bold=False):
        style = ParagraphStyle("text", fontName="BodyBold" if bold else "Body", fontSize=size,
                               leading=leading or size * 1.4, textColor=color, spaceAfter=0)
        p = Paragraph(escape(str(text)).replace("\n", "<br/>"), style)
        _, height = p.wrap(width, H)
        assert top - height >= 35, (self.p, "Paragraph overflow", str(text)[:90], top, height)
        p.drawOn(self.c, x, top - height)
        self.drawn.append({"page": self.p, "type": "text", "x": x, "bottom": top-height, "width": width, "height": height})
        return top - height

    def page(self, section, title, subtitle=""):
        if self.p: self.c.showPage()
        self.p += 1
        self.pages.append({"page":self.p,"section":section,"title":title})
        self.c.setFillColor(NAVY)
        self.c.rect(0, H-8, W, 8, fill=1, stroke=0)
        self.paragraph(section.upper(), M, H-23, CW, size=8.5, color=TEAL, bold=True)
        self.paragraph(title, M, H-42, CW, size=20, leading=24, bold=True)
        if subtitle: self.paragraph(subtitle, M, H-72, CW, size=9, color=GRAY)
        self.c.setStrokeColor(colors.HexColor("#d9e2e7"))
        self.c.line(M, 32, W-M, 32)
        self.c.setFont("Body", 8)
        self.c.setFillColor(GRAY)
        self.c.drawString(M, 19, "ABSA SỮA RỬA MẶT | BẢN NHÁP KIỂM TRA | 13/11/2026")
        self.c.drawRightString(W-M, 19, f"{self.p:02d}")
        self.c.bookmarkPage(f"page_{self.p}")
        self.c.addOutlineEntry(f"{self.p:02d}. {title}", f"page_{self.p}", level=0)
        return H-101

    def plot(self, name, x, top, width, max_height):
        path=FIGURES/f"{name}.png"
        with Image.open(path) as image: iw,ih=image.size
        scale=min(width/iw,max_height/ih)
        ww,hh=iw*scale,ih*scale
        assert top-hh>=35,(self.p,name,top,hh)
        self.c.drawImage(str(path),x+(width-ww)/2,top-hh,width=ww,height=hh,mask="auto")
        self.drawn.append({"page":self.p,"type":"figure","name":name,"x":x+(width-ww)/2,"bottom":top-hh,"width":ww,"height":hh})
        return top-hh

    def table(self, rows, headers, widths, x, top, size=9, highlights=None):
        style=ParagraphStyle("cell",fontName="Body",fontSize=size,leading=size*1.25,textColor=NAVY)
        hstyle=ParagraphStyle("head",fontName="BodyBold",fontSize=size,leading=size*1.25,textColor=colors.white)
        cells=[[Paragraph(escape(str(v)),hstyle) for v in headers]]
        cells += [[Paragraph(escape(str(v)),style) for v in row] for row in rows]
        t=Table(cells,colWidths=widths,hAlign="LEFT",repeatRows=1)
        commands=[("BACKGROUND",(0,0),(-1,0),NAVY),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                  ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
                  ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
                  ("LINEBELOW",(0,0),(-1,0),.7,NAVY)]
        for i in range(1,len(cells)):
            commands += [("BACKGROUND",(0,i),(-1,i),PALE if i%2 else colors.white),
                         ("LINEBELOW",(0,i),(-1,i),.3,colors.HexColor("#dce5eb"))]
        for i in highlights or []:commands.append(("BACKGROUND",(0,i+1),(-1,i+1),colors.HexColor("#dceee8")))
        t.setStyle(TableStyle(commands))
        _,height=t.wrap(sum(widths),H)
        assert top-height>=38,(self.p,"Table overflow",height,top,headers)
        t.drawOn(self.c,x,top-height)
        self.drawn.append({"page":self.p,"type":"table","x":x,"bottom":top-height,"width":sum(widths),"height":height})
        return top-height

    def caption(self,text,top):
        return self.paragraph(text,M,top-10,CW,size=8.6,leading=11.8,color=GRAY)

    def codes(self,top):
        text=" | ".join(f"A{i+1}: {NAMES[i]}" for i in range(10))
        return self.paragraph(text,M,top-8,CW,size=8.2,leading=11,color=GRAY)

    def save(self):
        self.c.save()
        (HERE/"report_layout.json").write_text(json.dumps({"pages":self.pages,"elements":self.drawn},ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    m=json.loads((HERE/"provenance.json").read_text(encoding="utf-8"))
    r=Report(); d=m["dataset"]; a=m["absa"]; ipa=data("25_ipa_overall")
    top=r.page("Báo cáo kết quả", "Nghiên cứu ABSA review sữa rửa mặt", "Bản nháp định lượng để kiểm tra nội dung, bảng và biểu đồ trước khi viết bài.")
    cards=[("REVIEW PHÂN TÍCH",fmt(d["valid_reviews"],0)),("TEXT DUY NHẤT",fmt(d["unique_texts"],0)),
           ("JOINT MACRO-F1 DẪN ĐẦU TEST",f"{a['joint_f1_pct']:.2f}%"),("ASPECT TRONG SHAP / IPA","10")]
    for i,(label,value) in enumerate(cards):
        x=M+i*(CW/4+1)
        r.c.setFillColor(PALE);r.c.roundRect(x,top-87,CW/4-9,85,7,fill=1,stroke=0)
        r.paragraph(label,x+12,top-13,CW/4-33,size=7.6,color=GRAY,bold=True)
        r.paragraph(value,x+12,top-38,CW/4-33,size=25,leading=30,color=TEAL,bold=True)
    top-=113
    r.paragraph("Các kết quả nổi bật",M,top,410,size=15,bold=True)
    r.paragraph(f"• Chuỗi tiền xử lý: {m['preprocessing']['raw_reviews']:,} review ban đầu, loại {m['preprocessing']['removed']:,}, còn {d['source_rows']:,} bản ghi nguồn; một bản ghi có text rỗng.\n"
                "• BiRNN + Focal Loss dẫn đầu trong 12 output test hiện có. So sánh CE-Focal được kiểm tra lại bằng suy luận và bootstrap ghép cặp.\n"
                "• Bao bì và dịch vụ shop thuộc nhóm ưu tiên cải thiện của IPA tổng thể theo ngưỡng mean.",M,top-30,445,size=11,leading=17)
    r.paragraph("Cấu trúc báo cáo",M+485,top,260,size=15,bold=True)
    r.paragraph("01  Tiền xử lý\n02  Tổng quan dữ liệu\n03  Aspect và sentiment\n04  Mô hình ABSA\n05  Rating và SHAP\n06  IPA và độ nhạy",M+485,top-30,260,size=11,leading=24)
    r.caption("Nguồn phân tích: dữ liệu gán nhãn hiện tại và các kết quả đánh giá đã lưu. Không huấn luyện lại ABSA. SHAP/IPA dùng nhãn tham chiếu; kết quả chưa phải đánh giá toàn bộ chuỗi tự động từ PhoBERT.",100)

    top=r.page("Phạm vi và cách đọc", "Ba đơn vị dữ liệu cần phân biệt", "Review nguồn, nội dung duy nhất và vị trí aspect không phải cùng một đơn vị đếm.")
    rows=[["Review thu thập",fmt(m["preprocessing"]["raw_reviews"],0),"Tổng review từ các CSV crawl"],
          ["Bản ghi nguồn sau các đợt dọn",fmt(d["source_rows"],0),"Giữ cả một cleaned_comment rỗng"],
          ["Review hợp lệ cho phân tích",fmt(d["valid_reviews"],0),"Mỗi dòng review đóng góp một lần"],
          ["Text duy nhất để huấn luyện ABSA",fmt(d["unique_texts"],0),"Không nhân theo duplicate_count"],
          ["Train / validation / test","8,882 / 1,904 / 1,904","Tách nhóm nội dung, tỷ lệ xấp xỉ 70/15/15"],
          ["Cặp nhãn gốc trong tập review",fmt(d["label_pairs"],0),"Một review có thể chứa nhiều cặp nhãn"]]
    bottom=r.table(rows,["Đơn vị / tập dữ liệu","Số lượng","Cách sử dụng"],[245,150,CW-395],M,top)
    r.paragraph("Quy ước cần giữ khi đọc kết quả",M,bottom-22,CW,size=13,bold=True)
    r.paragraph("Thống kê sentiment và mô hình ABSA giữ 11 aspect, gồm General. SHAP/IPA chỉ dùng 10 aspect cụ thể.\n"
                "Sentiment gốc có thể gồm nhiều cảm xúc trong cùng aspect; mô hình phân loại hiện giữ cảm xúc cuối, còn performance IPA loại các trường hợp nhiều cảm xúc.\n"
                "Các số thập phân dùng dấu chấm; dấu phẩy trong số nguyên phân tách hàng nghìn. Các biểu đồ được xuất riêng ở PNG 300 dpi và PDF.",M,bottom-47,CW,size=10,leading=16)

    top=r.page("01 / Tiền xử lý", "Số lượng dữ liệu qua từng bước", "Các bước xóa được phân biệt giữa bằng chứng nhật ký và chênh lệch khôi phục từ snapshot.")
    bottom=r.plot("01_preprocessing_flow",M,top,CW,175)
    hist=data("01_preprocessing_stages")
    rows=[[x.Stage,fmt(x.Before,0),fmt(x.Removed,0),fmt(x.After,0),x.Evidence] for x in hist.itertuples()]
    bottom=r.table(rows,["Bước","Trước","Loại","Sau","Bằng chứng"],[265,75,55,75,CW-470],M,bottom-10,size=8.3)
    r.caption("248 trường hợp nội dung gồm 175 không phù hợp, 72 nội dung trộn và 1 chưa rõ; trong đó có 13 mẫu review trống nội dung. Bước nhãn rỗng loại 17 text duy nhất, tương ứng 19 review. Cắt phần trả lời shop nhưng giữ nội dung khách hàng ở 1,659 review, không tính là xóa review.",bottom)

    top=r.page("02 / Tổng quan dữ liệu", "Quy mô theo nền tảng và thương hiệu", "Thống kê trên 16,292 review có cleaned_comment và nhãn; gộp tên chỉ khác hoa/thường hoặc khoảng trắng.")
    bottom=r.plot("02_platform_and_top_brands",M,top,CW,280)
    pf=data("02_platform_overview")
    rows=[[x.Platform,fmt(x.Reviews,0),fmt(x.Share_Pct,1),fmt(x.Products,0),fmt(x.Brands,0),fmt(x.Mean_Rating)] for x in pf.itertuples()]
    bottom=r.table(rows,["Nền tảng","Review","Tỷ trọng (%)","Sản phẩm","Thương hiệu","Rating TB"],[140,125,135,120,125,CW-645],M,bottom-8,size=9)
    r.caption(f"Toàn bộ tập phân tích có {d['brands']} tên thương hiệu sau chuẩn hóa và {d['products']} URL sản phẩm. URL được đếm như định danh sản phẩm trong từng nguồn, chưa hợp nhất sản phẩm giống nhau giữa các sàn.",bottom)

    top=r.page("02 / Tổng quan dữ liệu", "Phân phối độ dài review", "Độ dài ở trang này là số đơn vị cách nhau bằng khoảng trắng trong cleaned_comment, chưa phải token PhoBERT.")
    bottom=r.plot("03_review_lengths",M,top,CW,260)
    lengths=data("04_length_summary")
    rows=[[x.Group,fmt(x.N,0),fmt(x.Mean),fmt(x.P50,0),fmt(x.P95,0),fmt(x.P99,0),fmt(x.Max,0)] for x in lengths.itertuples()]
    bottom=r.table(rows,["Nhóm","N","Mean","P50","P95","P99","Max"],[180,100,100,95,95,95,CW-665],M,bottom-8,size=8.8)
    r.caption("Histogram phóng tới P99.5 (tối đa 250 đơn vị); bảng vẫn giữ toàn bộ đuôi phân phối. Boxplot ẩn điểm ngoại lệ để dễ so sánh, không xóa review. Độ dài phân bố lệch phải nên báo cáo cả median và các phân vị.",bottom)

    top=r.page("02 / Tổng quan dữ liệu", "Token PhoBERT và mức cắt ở 128", "Đếm trực tiếp bằng vinai/phobert-base từ cache cục bộ; gồm token đặc biệt, không cắt trước khi thống kê.")
    bottom=r.plot("08_phobert_token_distribution",M,top,CW,240)
    tokens=data("09_phobert_token_lengths")
    rows=[[x.Dataset,fmt(x.N,0),fmt(x.Mean),fmt(x.P95,0),fmt(x.P99,0),fmt(x.Max,0),fmt(x.Over_128,0),fmt(x.Over_128_Pct)] for x in tokens.itertuples()]
    bottom=r.table(rows,["Tập","N","Mean","P95","P99","Max",">128",">128 (%)"],[155,95,85,80,80,80,90,CW-665],M,bottom-8,size=8.3)
    r.caption("Tỷ lệ >128 cho biết số mẫu sẽ bị cắt ít nhất một token trong thiết lập training/evaluation hiện tại. Biểu đồ có thể phóng phần đầu phân phối; Max trong bảng được tính trên đầy đủ dữ liệu.",bottom)

    top=r.page("02 / Tổng quan dữ liệu", "Phân phối rating giữa các nền tảng", "Rating là biến đích của mô hình hồi quy dùng cho SHAP; không được dùng để suy ra nhãn sentiment.")
    bottom=r.plot("04_rating_by_platform",M,top,CW,245)
    counts=data("05_rating_counts")
    rows=[[row.iloc[0]]+[fmt(v,0) for v in row.iloc[1:]] for _,row in counts.iterrows()]
    bottom=r.table(rows,["Nền tảng","1 sao","2 sao","3 sao","4 sao","5 sao"],[155]+[(CW-155)/5]*5,M,bottom-12,size=9)
    r.paragraph("Cách diễn giải",M,bottom-20,CW,size=12,bold=True)
    r.paragraph("Khác biệt rating cần được đặt cạnh khác biệt sentiment và IPA. Crawler Tiki có quota theo số sao, vì vậy cơ cấu rating của tập này phản ánh cách lấy mẫu và không thể coi là cơ cấu của toàn bộ thị trường. Các so sánh trong bản nháp mô tả tập review đã thu thập.",M,bottom-43,CW,size=10.3,leading=15)

    top=r.page("03 / Aspect và sentiment", "Phân phối cảm xúc trên 11 aspect", "Dùng nhãn gốc, đếm mỗi cặp aspect-sentiment khác nhau tối đa một lần trong một review.")
    bottom=r.plot("05_sentiment_counts_and_shares",M,top,CW,350)
    totals=d["sentiment_totals"]
    r.caption(f"Tổng {d['label_pairs']:,} cặp nhãn: positive {totals['positive']:,}, negative {totals['negative']:,}, neutral {totals['neutral']:,}. Phần trăm cột chồng dùng mẫu số là tổng cặp nhãn của từng aspect; không phải số review. Có {d['multi_sentiment_reviews']:,} review mang nhiều sentiment trong ít nhất một aspect.",bottom)

    top=r.page("03 / Aspect và sentiment", "Số lượng tiêu cực và tỷ lệ tiêu cực", "Hai thước đo trả lời hai câu hỏi khác nhau: khối lượng phản ánh tiêu cực và mức tập trung tiêu cực trong một aspect.")
    bottom=r.plot("06_negative_counts_and_rates",M,top,CW,350)
    r.caption("Negative rate = số review có nhãn negative cho aspect / số review đề cập aspect. Các trường hợp vừa positive vừa negative vẫn được tính là có negative trong thống kê mô tả này. Quy tắc đó khác performance IPA, vốn loại trường hợp nhiều sentiment khỏi mẫu số.",bottom)

    top=r.page("03 / Aspect và sentiment", "Tiêu cực theo sàn và top 5 thương hiệu", "So sánh tỷ lệ có nhãn negative trong từng nhóm; mỗi ô dùng mẫu số review đề cập aspect của chính nhóm đó.")
    bottom=r.plot("07_negative_rate_by_group",M,top,CW,345)
    r.caption("Ký hiệu *: dưới 30 review đề cập aspect. Các ô màu có thể cùng mức trần 50% để so sánh; số trong ô vẫn giữ giá trị thật. Tỷ lệ của nhóm ít mẫu cần được đọc cùng số lượng trong bảng sentiment_by_group, không suy ra khác biệt có ý nghĩa thống kê từ màu sắc.",bottom)

    top=r.page("03 / Aspect và sentiment", "Cơ cấu ba sentiment theo sàn và thương hiệu", "Tổng hợp positive, negative và neutral trên 11 aspect; mỗi nhóm có mẫu số là tổng số cặp nhãn của chính nhóm đó.")
    bottom=r.plot("07b_group_sentiment_overall",M,top,CW,325)
    r.caption("n trong nhãn trục là số review của nhóm, không phải mẫu số của các tỷ lệ trên thanh. Một review có thể đóng góp nhiều cặp aspect-sentiment; các tỷ lệ thể hiện cơ cấu cặp nhãn. Phần thanh dưới 10% được lược chữ để tránh chồng nhãn. CSV giữ đầy đủ số đếm và tỷ lệ; không gán một sentiment chung duy nhất cho toàn bộ review.",bottom)

    top=r.page("04 / Mô hình ABSA", "Kết quả tổng quan của 12 cấu hình", "Số liệu từ các output đánh giá đã lưu, cùng 1,904 text test và cùng evaluator.")
    metrics=data("11_absa_model_metrics")
    rows=[]
    for (model,scenario),sub in metrics.groupby(["Model","Scenario"],sort=False):
        t=sub.set_index("Task")
        rows.append([MODEL_NAMES[model],SCENARIOS[scenario],fmt(t.loc["ACD","F1_Macro"]),fmt(t.loc["SPC","F1_Macro"]),fmt(t.loc["ACD+SPC","F1_Macro"])])
    bottom=r.table(rows,["Kiến trúc PhoBERT","Loss","ACD macro-F1 (%)","SPC macro-F1 (%)","Joint macro-F1 (%)"],[175,145,150,150,CW-620],M,top,size=9,highlights=[i for i,row in enumerate(rows) if row[0]=="BiRNN" and row[1]=="Focal Loss"])
    ranking=data("14_absa_ranking").head(3)
    top_text="; ".join(f"{MODEL_NAMES[x.Model]} / {SCENARIOS[x.Scenario]}: {x.F1_Macro:.2f}%" for x in ranking.itertuples())
    r.caption("Ba cấu hình dẫn đầu joint macro-F1 test: "+top_text+". Đây là xếp hạng mô tả; việc chọn mô hình nên dựa trên validation. Chênh lệch nhỏ khoảng 0.2 điểm phần trăm giữa hai bản đầu chưa chứng minh khác biệt có ý nghĩa thống kê.",bottom)

    top=r.page("04 / Mô hình ABSA", "So sánh CE, CE trọng số và Focal Loss", "So sánh loss trong cùng kiến trúc để tránh trộn ảnh hưởng kiến trúc và ảnh hưởng hàm mất mát.")
    bottom=r.plot("09_absa_architecture_loss",M,top,CW,275)
    boot=data("16_paired_bootstrap")
    rows=[[x.Model.replace("birnn_","BiRNN / "),fmt(x.Joint_F1_Pct),f"[{x.CI95_Lower:.2f}; {x.CI95_Upper:.2f}]"] for x in boot.itertuples()]
    bottom=r.table(rows,["Checkpoint","Joint macro-F1 (%)","Khoảng tin cậy bootstrap 95%"],[270,200,CW-470],M,bottom-8,size=9)
    b=a["paired_bootstrap"]
    r.caption(f"Suy luận lại hai checkpoint khớp output gốc. Chênh lệch Focal - CE = {b['difference_focal_minus_ce_pp']:.2f} điểm %, CI 95% [{b['ci95_lower']:.2f}; {b['ci95_upper']:.2f}], từ 1,000 lần lấy mẫu ghép cặp review test. Khoảng này chỉ phản ánh lấy mẫu test, chưa phản ánh biến thiên do huấn luyện nhiều seed.",bottom)

    top=r.page("04 / Mô hình ABSA", "F1 theo aspect: BiRNN Focal so với CE", "Cùng kiến trúc và cùng tập test; F1 từng aspect là trung bình F1 của ba cặp sentiment theo evaluator.")
    bottom=r.plot("10_absa_aspect_focal_vs_ce",M,top,CW,345)
    comp=data("15_leading_architecture_aspect_comparison")
    largest=comp.sort_values("Delta_Focal_CE",ascending=False).head(3)
    r.caption("Các mức tăng lớn nhất trong kiến trúc BiRNN: "+"; ".join(f"{LABELS[x.Aspect]} +{x.Delta_Focal_CE:.2f} điểm %" for x in largest.itertuples())+". Bảng đầy đủ và support nằm ở phụ lục của báo cáo. Điểm tổng thể cao hơn không bảo đảm mọi aspect đều cao hơn.",bottom)

    top=r.page("04 / Mô hình ABSA", "Hiệu năng từng sentiment và nhãn hiếm", "BiRNN + Focal Loss: F1 và support của 33 cặp aspect-sentiment.")
    bottom=r.plot("11_absa_sentiment_f1_support",M,top,CW,350)
    r.caption("Các ô F1 phải được đọc cùng support. Ví dụ một số neutral hoặc negative chỉ có vài mẫu test; F1 bằng 0 không đồng nghĩa không có đánh giá về aspect. Chênh lệch giữa ACD cao và joint F1 thấp phản ánh đây là các nhiệm vụ và cách lấy trung bình khác nhau.",bottom)

    top=r.page("04 / Mô hình ABSA", "Ma trận nhầm lẫn của checkpoint dẫn đầu", "Tính lại trên 1,904 text × 11 aspect = 20,944 vị trí; None là trạng thái nội bộ khi aspect không được đề cập.")
    bottom=r.plot("12_best_model_confusion",M,top,CW,270)
    err=data("20_error_type_counts")
    bottom=r.table([[x.Error_Type,fmt(x.Count,0)] for x in err.itertuples()],["Nhóm lỗi ở vị trí aspect","Số vị trí"],[CW*.7,CW*.3],M,bottom-8,size=9)
    r.caption("Một review có thể đóng góp nhiều vị trí lỗi. Ma trận gộp giúp chẩn đoán nhầm lẫn; không thay thế macro-F1 33 cặp. Cột None dự đoán biểu thị bỏ sót aspect, hàng None thật biểu thị phát hiện dư aspect.",bottom)

    top=r.page("04 / Mô hình ABSA", "Ví dụ lỗi từ suy luận thực tế", "Chọn hai ví dụ ngắn cho mỗi loại lỗi; đây là mẫu minh họa có chủ đích, không phải mẫu ngẫu nhiên đại diện.")
    examples=data("19_error_examples")
    rows=[[f"{x.Test_Row_1based}\n{LABELS[x.Aspect]}",x.Review,f"{x.Gold}\n→ {x.Predicted}",x.Error_Type] for x in examples.itertuples()]
    bottom=r.table(rows,["Dòng test / aspect","cleaned_comment","Tham chiếu → dự đoán","Loại lỗi"],[150,365,130,CW-645],M,top,size=9)
    r.caption("Tham chiếu ở đây tuân thủ target hiện tại: nếu cùng aspect có nhiều sentiment thì giữ sentiment xuất hiện cuối. Loại lỗi được xác định bằng so sánh target và dự đoán; chưa kết luận nguyên nhân ngôn ngữ hoặc nhãn tham chiếu sai. Tệp best_model_errors lưu toàn bộ các vị trí lỗi để duyệt tiếp.",bottom)

    top=r.page("05 / Rating và SHAP", "Đánh giá mô hình dự đoán số sao", "Đây là nhiệm vụ hồi quy rating, tách biệt với mô hình PhoBERT phân loại aspect-sentiment.")
    regression=data("22_rating_regression_oof")
    rows=[[x.Model,fmt(x.RMSE,4),fmt(x.MAE,4),fmt(x.R2,4)] for x in regression.itertuples()]
    bottom=r.table(rows,["Mô hình","OOF RMSE","OOF MAE","OOF R²"],[275,170,170,CW-615],M,top,size=11)
    folds=data("23_rating_regression_folds")
    rows=[]
    for model,sub in folds.groupby("Model"):
        rows.append([model]+[f"{sub[col].mean():.4f} ± {sub[col].std():.4f}" for col in ["RMSE","MAE","R2"]])
    bottom=r.table(rows,["Mô hình","RMSE: mean ± SD","MAE: mean ± SD","R²: mean ± SD"],[235,180,180,CW-595],M,bottom-20,size=9)
    r.paragraph("Phạm vi giải thích",M,bottom-22,CW,size=13,bold=True)
    r.paragraph("XGBoost được dùng cho native Tree SHAP và có OOF RMSE thấp nhất trong ba kết quả hiện có; SVR có MAE thấp hơn. Đầu vào gồm 20 biến sentiment của 10 aspect và hai biến giá/giảm giá.\n"
                "Các metric trên là kết quả KFold 10 fold thông thường đã lưu. Mô hình giải thích được fit lại trên toàn bộ dữ liệu sau đánh giá. SHAP của mô hình toàn bộ không phải SHAP ngoài fold.",M,bottom-46,CW,size=10.3,leading=16)

    top=r.page("05 / Rating và SHAP", "Importance và tần suất đề cập aspect", "Importance = mean(|SHAP_Pos| + |SHAP_Neg|), sau đó chuẩn hóa trên 10 aspect. General bị loại khỏi đầu vào.")
    bottom=r.plot("13_shap_importance_and_frequency",M,top,CW,335)
    bottom=r.codes(bottom)
    r.caption("SHAP được tính trên toàn bộ review hợp lệ, gồm cả review không đề cập aspect. Giá và giảm giá có trong mô hình nhưng không nằm trong mẫu số chuẩn hóa importance của 10 aspect. Tần suất đề cập và importance là hai đại lượng khác nhau; mỗi điểm là một aspect.",bottom)

    top=r.page("05 / Rating và SHAP", "Đóng góp positive và negative theo aspect", "Mean và median của signed SHAP chỉ tính trên review có biến sentiment tương ứng bằng 1.")
    bottom=r.plot("14_shap_signed_mean_median",M,top,CW,345)
    r.caption("Dấu SHAP biểu thị chiều đóng góp vào rating dự đoán so với baseline của mô hình. Những review có nhiều sentiment vẫn có thể bật cả hai biến. Đây là giải thích dự đoán, không phải tác động nhân quả. Mean/median ở trang này không thay đổi công thức importance và performance IPA.",bottom)

    top=r.page("05 / Rating và SHAP", "Tương tác giữa các aspect khác nhau", "Tính lại bằng cùng tiền xử lý và XGBoost của IPA; không sử dụng đầu ra từ bộ đọc giá/discount cũ của KANO.")
    bottom=r.plot("15_top_cross_aspect_interactions",M,top,CW,300)
    bottom=r.codes(bottom)
    r.caption("Mỗi cặp dùng 2 × SHAP interaction ở ngoài đường chéo, rồi lấy trung bình trị tuyệt đối trên toàn bộ review. Chỉ xếp hạng cặp khác aspect có ít nhất 20 review cùng bật hai biến; n là số review đồng kích hoạt. Dấu (+)/(-) là nhãn Pos/Neg của biến, không phải dấu của interaction. Các cặp Pos-Neg cùng aspect được giữ trong CSV nhưng tách khỏi hình này.",bottom)

    top=r.page("05 / Rating và SHAP", "Hai giải thích cục bộ cho rating dự đoán", "Minh họa review 1 sao và 5 sao có dự đoán gần rating thật, với text ngắn; không đại diện mọi review.")
    bottom=r.plot("16_local_rating_explanations",M,top,CW,280)
    examples=data("29_local_rating_examples")
    for i,x in enumerate(examples.itertuples()):
        r.paragraph(f"Ví dụ {i+1} - dòng hợp lệ {x.Valid_Row_0based} (đếm từ 0): {x.Review}",M+i*(CW/2+8),bottom-12,CW/2-16,size=9.2,leading=13)
    r.caption("Mỗi waterfall bắt đầu từ baseline rồi cộng sáu đóng góp lớn nhất và tổng các biến còn lại; tổng khớp dự đoán với sai số dưới 0.001. Đóng góp của biến không được bật vẫn có thể khác 0 trong SHAP. Đây là mô hình rating dùng nhãn tham chiếu, không phải giải thích token của PhoBERT.",110)

    top=r.page("06 / IPA", "IPA tổng thể: các khía cạnh cần ưu tiên", "Mean cuts: importance = 10.000%; performance = 81.594/100.")
    r.plot("17_ipa_overall",M,top,455,330)
    rows=[[f"A{ASPECTS.index(x.Aspect)+1}",fmt(x.Importance_Pct),fmt(x.Performance_0_100),QSHORT[x.IPA_Quadrant]] for x in ipa.itertuples()]
    r.table(rows,["Aspect","I (%)","P /100","Quadrant"],[47,59,65,CW-455-16-171],M+471,top,size=8.5)
    bottom=r.codes(top-340)
    r.caption("Performance = 100 × (N_positive + 0.5 × N_neutral) / (N_positive + N_neutral + N_negative). Review không đề cập và trường hợp nhiều sentiment cùng aspect bị loại khỏi điểm này. Bao bì và dịch vụ shop có importance cao nhưng performance dưới ngưỡng mean, thuộc nhóm ưu tiên.",bottom)

    top=r.page("06 / IPA", "IPA theo từng nền tảng", "Mỗi nền tảng có mô hình, phân phối SHAP và hai ngưỡng riêng; importance được chuẩn hóa trong chính nền tảng đó.")
    bottom=r.plot("18_ipa_platform_matrices",M,top,CW,245)
    summary=data("32_ipa_platform_summary")
    rows=[[x.Platform,fmt(x.Review_N,0),fmt(x.Importance_Cut),fmt(x.Performance_Cut),fmt(x.XGBoost_R2,3)] for x in summary.itertuples()]
    bottom=r.table(rows,["Nền tảng","N","Ngưỡng I (%)","Ngưỡng P","OOF R² XGBoost"],[180,125,155,155,CW-615],M,bottom-6,size=8.8)
    bottom=r.codes(bottom)
    r.caption("Các đường cắt là giá trị tương đối. Một aspect ở quadrant tốt hơn trên sàn A không tự chứng minh chất lượng tuyệt đối cao hơn sàn B. Các figure riêng từng sàn được xuất cùng bộ kết quả để phóng lớn khi cần.",bottom)

    top=r.page("06 / IPA", "So sánh nhóm ưu tiên giữa sàn và thương hiệu", "Các ô hiển thị quadrant, không phải thứ hạng chất lượng tuyệt đối giữa các nhóm.")
    bottom=r.plot("20_ipa_quadrant_comparison",M,top,CW,345)
    r.caption("Ưu tiên = Concentrate Here; Duy trì = Keep Up the Good Work; Theo dõi = Low Priority; Cân nhắc = Possible Overkill. Mỗi nhóm dùng cutoffs và mô hình riêng. Cần đọc ô kết quả cùng số review và số lượng đề cập aspect; các nhóm nhiều/ít review không có cùng độ chắc chắn.",bottom)

    top=r.page("06 / IPA", "IPA của năm thương hiệu nhiều review nhất", "Chọn top 5 theo số review hợp lệ sau chuẩn hóa tên; mỗi thương hiệu được phân tích độc lập.")
    bottom=r.plot("19_ipa_brand_matrices",M,top,CW,335)
    bottom=r.codes(bottom)
    r.caption("N: Hada Labo 1,525; La Roche-Posay 1,150; Cetaphil 1,112; Acnes 1,025; CeraVe 799. Một số aspect có ít hơn 30 lượt sentiment hợp lệ, đặc biệt ở Acnes và thành phần/công thức. Figure riêng từng thương hiệu và toàn bộ bảng support được lưu để kiểm tra chi tiết.",bottom)

    top=r.page("06 / IPA", "Độ nhạy khi dùng ngưỡng mean và median", "Giữ nguyên importance và performance; chỉ thay cách tính hai đường phân chia.")
    bottom=r.plot("21_ipa_mean_median",M,top,CW,260)
    thresholds=data("35_ipa_cutoff_comparison")
    rows=[[x.Cut_Method,fmt(x.Importance_Cut,3),fmt(x.Performance_Cut,3)] for x in thresholds.itertuples()]
    bottom=r.table(rows,["Cách lấy ngưỡng","Importance (%)","Performance /100"],[280,245,CW-525],M,bottom-5,size=9)
    changes=data("34_ipa_mean_median_comparison")
    changed=changes.loc[changes.Quadrant_Changed]
    text="; ".join(f"{LABELS[x.Aspect]}: {QSHORT[x.IPA_Quadrant_Mean]} → {QSHORT[x.IPA_Quadrant_Median]}" for x in changed.itertuples())
    r.caption(f"{len(changed)}/10 aspect đổi quadrant: {text}. Mean và median ở đây áp dụng cho cutoffs, không thay mean |SHAP| bằng median |SHAP|.",bottom)

    top=r.page("06 / IPA", "Bảng tổng hợp ưu tiên theo nhóm", "Đọc theo hàng để tìm tính nhất quán của ưu tiên; đọc theo cột để xây dựng hàm ý cho từng nhóm.")
    priority=data("36_ipa_priority_comparison")
    cols=priority.columns.tolist()
    rows=[[LABELS[row[0]]]+[QSHORT[v] for v in row[1:]] for row in priority.itertuples(index=False,name=None)]
    headers=["Aspect"]+["Tổng thể" if c=="Overall" else c for c in cols[1:]]
    bottom=r.table(rows,headers,[160]+[(CW-160)/(len(headers)-1)]*(len(headers)-1),M,top,size=8.0)
    summary=data("33_ipa_brand_summary")
    r.caption("Các nhãn được rút gọn như trang so sánh quadrant. Các bảng IPA gốc có importance, performance, số mẫu hợp lệ và số trường hợp nhiều sentiment bị loại. Possible Overkill chỉ là gợi ý xem xét phân bổ nguồn lực; không phải bằng chứng nên giảm chất lượng của aspect.",bottom)

    top=r.page("Phụ lục / ABSA", "Bảng chi tiết theo aspect của BiRNN", "Giữ nguyên cách đánh giá hiện tại; support là số review test có aspect sau khi tạo target.")
    comp=data("15_leading_architecture_aspect_comparison")
    rows=[[LABELS[x.Aspect],fmt(x.baseline),fmt(x.class_weights),fmt(x.focal_loss),f"{x.Delta_Focal_CE:+.2f}",fmt(x.Support,0)] for x in comp.itertuples()]
    bottom=r.table(rows,["Aspect","CE F1 (%)","CE trọng số","Focal F1 (%)","Focal - CE (điểm %)","Support"],[235,100,110,110,145,CW-700],M,top,size=9)
    r.caption("Đây là so sánh loss trong cùng kiến trúc BiRNN. Các kết quả đầy đủ của bốn kiến trúc và 33 cặp sentiment nằm trong các CSV của bộ kết quả. Không lấy mô hình tệ nhất ở kiến trúc khác làm đối chứng duy nhất cho hiệu quả Focal Loss.",bottom)

    top=r.page("Giới hạn và việc cần kiểm chứng", "Các kết quả tạm thời và phạm vi kết luận", "Những điểm dưới đây giúp xác định phần đã có bằng chứng và phần cần thí nghiệm hoặc duyệt thêm.")
    rows=[
        ["Nhãn tham chiếu và chuỗi tự động","SHAP/IPA dùng label đã gán, chưa dùng đầu ra PhoBERT. Chưa thể coi là kiểm chứng end-to-end của toàn bộ hệ thống."],
        ["Trường hợp nhiều sentiment","Nhãn gốc cho phép nhiều sentiment; target ABSA giữ cuối, còn IPA performance loại các trường hợp này. Cần nêu rõ trong phương pháp."],
        ["Độ ổn định mô hình ABSA","Bootstrap đã thực hiện trên hai checkpoint cố định; chưa có nhiều seed huấn luyện. Chênh lệch khoảng 0.2 điểm % giữa hai bản dẫn đầu cần thận trọng."],
        ["Độ khái quát hồi quy","CV hiện tại là KFold thông thường trên review có text lặp. Nên kiểm tra thêm theo nhóm text; theo nhóm sản phẩm nếu mục tiêu là sản phẩm chưa gặp."],
        ["Thiết kế lấy mẫu","Tiki có quota rating; toàn bộ kết quả mô tả corpus thu thập, không đại diện mặc định cho toàn bộ thị trường hoặc mọi người mua."],
        ["Nhóm ít mẫu","Một số cặp aspect-sentiment và một số aspect theo thương hiệu có support thấp. Số liệu mô tả chưa phải kiểm định khác biệt giữa các nhóm."],
        ["Cách hiểu SHAP và IPA","Giải thích đóng góp trong dự đoán, không chứng minh quan hệ nhân quả hay tác động của một biện pháp cải thiện thực tế."],
        ["Kiểm tra chất lượng nhãn","Chưa có đánh giá độ đồng thuận của nhiều người gán nhãn độc lập trong bộ kết quả này. Ví dụ lỗi cần được duyệt lại theo ngữ cảnh."],
    ]
    bottom=r.table(rows,["Vấn đề","Phạm vi hiện tại / bước kiểm chứng"],[195,CW-195],M,top,size=9.3)
    r.caption("Bản nháp giữ các kết quả hiện có, không thay đổi mô hình, loss, evaluator, split hoặc CSV nguồn. Riêng interaction được tính lại bằng canonical IPA preprocessing để nhất quán trong báo cáo.",bottom)

    top=r.page("Nguồn và tái lập", "Dữ liệu nguồn, kiểm tra và tài liệu tham chiếu", "SHA-256 đầy đủ của mọi đầu vào được lưu trong provenance.json cạnh mã tạo báo cáo.")
    wanted=["crawl-data/data/combined_reviews_brand_filled_language_filtered.csv","Labeling/data/unique_texts_gemini_labeled.csv",
            "train-models/data/train_cleanser.csv","train-models/data/val_cleanser.csv","train-models/data/test_cleanser.csv",
            "evaluation/evaluation_2_cleanser.py","Business/ipa_shap_cleanser.py","json_cleanup_backup.zip","data_cleanup_backup.zip"]
    rows=[[name,m["inputs"][name][:20]+"..."] for name in wanted]
    bottom=r.table(rows,["Đầu vào","SHA-256 (rút gọn)"],[540,CW-540],M,top,size=8.6)
    r.paragraph(f"Đã đối chiếu nguồn không đổi: {m['validation']['source_files_unchanged']} tệp. Đã kiểm tra cộng SHAP bằng dự đoán, cộng interaction bằng SHAP, suy luận ABSA khớp output, split không giao text và importance tổng bằng 100%.\n"
                "Tham khảo cách diễn giải: tài liệu SHAP về predictive models và causal insights; tài liệu scikit-learn về cross-validation và GroupKFold; Joung & Kim (2021), Approach for Importance-Performance Analysis of Product Attributes From Online Reviews, DOI 10.1115/1.4049865.\n"
                "Bảng và biểu đồ độc lập được lưu trong tables/ và figures/. Hướng dẫn tái tạo có trong README.md. Báo cáo này phục vụ kiểm tra nội dung trước khi chọn các hình/bảng đưa vào bản thảo chính.",M,bottom-20,CW,size=9.5,leading=14)
    r.save()
    print(f"PDF written: {OUTPUT} ({r.p} pages)",flush=True)


if __name__=="__main__":main()
