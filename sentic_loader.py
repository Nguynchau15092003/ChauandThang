# sentic_loader.py

import os

_sentic_cache = {}  # Dùng để cache từ điển

def load_senticnet(opt):
    """
    Tải SenticNet dựa trên ngôn ngữ, cache lại để tránh load nhiều lần.
    """
    lang = getattr(opt, 'sentic', 'en')
    
    if lang in _sentic_cache:
        return _sentic_cache[lang]

    if lang == 'vi':
        path = './Sentic/senticnet_vi/senticnet_vi.txt'
        print("[SenticNet] Using Vietnamese SenticNet")
    else:
        path = './Sentic/senticnet/senticnet.txt'
        print("[SenticNet] Using English SenticNet")

    sentic_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                concept = parts[0].lower()
                try:
                    polarity = float(parts[1])
                    sentic_dict[concept] = polarity
                except ValueError:
                    continue

    _sentic_cache[lang] = sentic_dict
    return sentic_dict