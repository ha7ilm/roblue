#!/bin/sh

python3 sympybotics_to_torch.py sympybotexp_C_code.py sympybotconv_C_code.py
python3 sympybotics_to_torch.py sympybotexp_M_code.py sympybotconv_M_code.py
python3 sympybotics_to_torch.py sympybotexp_g_code.py sympybotconv_g_code.py