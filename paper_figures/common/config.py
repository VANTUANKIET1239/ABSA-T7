"""Shared presentation conventions; original ABSA labels are never renamed."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parent
DATA = BASE / 'data'
FIGURES = BASE / 'output'
TABLES = BASE / 'tables'
ASPECTS = ['CleansingEffectiveness','HydrationSkinComfort','SkinCompatibilitySafety',
           'TextureFoaming','Fragrance','IngredientsFormulation','Packaging','PriceValue',
           'ShopService','Delivery','General']
NAMES = ['Hiệu quả làm sạch','Cấp ẩm và độ dễ chịu','Mức độ an toàn','Kết cấu và tạo bọt',
         'Mùi hương','Thành phần','Bao bì','Giá cả','Dịch vụ cửa hàng','Giao hàng','Đánh giá tổng quan']
CODES = dict(zip(ASPECTS, [f'A{i}' for i in range(1,12)]))
LABELS = dict(zip(ASPECTS,NAMES))
ASPECT_CAPTIONS = {aspect: f'{LABELS[aspect]} - {CODES[aspect]}' for aspect in ASPECTS}
PLATFORMS = ['Lazada','Shopee','Tiki']
BRANDS = ['Hada Labo','La Roche-Posay','Cetaphil','Acnes','CeraVe']
MODEL_NAMES = {'birnn':'PhoBERT + BiRNN','bilstm':'PhoBERT + BiLSTM',
               'bigru':'PhoBERT + BiGRU','multitask':'PhoBERT MultiTask'}
LOSSES = {'baseline':'CE','class_weights':'WCE','focal_loss':'Focal Loss'}
QUADRANTS = ['Concentrate Here','Keep Up the Good Work','Low Priority','Possible Overkill']
Q_NAMES = dict(zip(QUADRANTS,['Ưu tiên cải thiện','Tiếp tục duy trì','Ưu tiên thấp','Cân nhắc phân bổ nguồn lực']))
Q_COLORS = ['#b95450','#31846c','#a6a6a6','#5481ad']

