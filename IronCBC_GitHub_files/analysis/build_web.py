"""Embed model/ironcbc_model.json into web/index.html (run after re-exporting the model)."""
import os
H = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(H, 'app_template.html'), encoding='utf-8').read()
m = open(os.path.join(H, '..', 'model', 'ironcbc_model.json'), encoding='utf-8').read()
open(os.path.join(H, '..', 'web', 'index.html'), 'w', encoding='utf-8').write(s.replace('__MODEL_JSON__', m))
print('web/index.html written')
