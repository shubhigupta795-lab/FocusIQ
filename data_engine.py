"""Load the exact uploaded CSV; no generated training rows."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).parent
DATA = ROOT / 'data'
SOURCE = DATA / 'student_productivity_distraction_dataset_20000.csv'
AGE = ['17–20', '21–25', '26–29']
CLASS_NAMES = ['Low', 'Normal', 'High']
STYLE = ['All alerts', 'Important only', 'Mostly muted']
NUM = [
    'screen_hours',
    'social_hours',
    'video_hours',
    'gaming_hours',
    'study_hours',
    'sleep_hours',
    'breaks',
    'exercise',
    'stress',
    'age'
]
CAT = ['gender']
FEATURES = NUM + CAT
MAP = {
    'student_id': 'user_id',
    'phone_usage_hours': 'screen_hours',
    'social_media_hours': 'social_hours',
    'youtube_hours': 'video_hours',
    'study_hours_per_day': 'study_hours',
    'breaks_per_day': 'breaks',
    'exercise_minutes': 'exercise',
    'stress_level': 'stress',
    'productivity_score': 'productivity'
}

def engineer(d):
    d = d.copy()
    d['distraction_index'] = 100 * (0.5 * np.clip(d.screen_hours / 12, 0, 1) + 0.5 * np.clip(d.social_hours / 8, 0, 1))
    return d

def load():
    raw = pd.read_csv(SOURCE)
    d = raw.rename(columns=MAP).copy()
    d = d.drop_duplicates('user_id')
    d = d.dropna(subset=['user_id', 'productivity'])
    d = d[d.productivity.between(0, 100)].copy()
    d['age_group'] = pd.cut(d.age, [16, 20, 25, 29], labels=AGE).astype('object')
    d = engineer(d)
    d['low_focus'] = (d.productivity < 40).astype(int)
    d['productivity_class'] = np.select(
        [d.productivity < 40, d.productivity < 70], [0, 1], default=2
    ).astype(int)
    return (
        d,
        {
            'source_rows': len(raw),
            'duplicate_ids_removed': int(raw.student_id.duplicated().sum()),
            'clean_rows': len(d),
            'missing_source_cells': int(raw.isna().sum().sum()),
            'category_sum_exceeds_phone_hours': int((d.social_hours + d.video_hours + d.gaming_hours > d.screen_hours).sum())
        }
    )

def wellness(h):
    parts = [
        (0.3, 1 - min(h['screen_hours'] / 12, 1)),
        (0.3, max(0, 1 - abs(h['sleep_hours'] - 8) / 4))
    ]
    if h.get('notifications') is not None:
        parts.append((0.2, 1 - min(h['notifications'] / 200, 1)))
    if h.get('night_minutes') is not None:
        parts.append((0.2, 1 - min(h['night_minutes'] / 180, 1)))
    return round(100 * sum((w * v for w, v in parts)) / sum((w for w, v in parts)))

def persona(h):
    if (h.get('night_minutes') or 0) >= 90:
        return 'Nighttime scroller'
    if (h.get('notifications') or 0) >= 110 or (h.get('checks') or 0) >= 100:
        return 'Notification magnet'
    biggest = max(['social', 'video', 'gaming'], key=lambda k: h[k + '_hours'])
    return {
        'social': 'Social explorer',
        'video': 'Video enthusiast',
        'gaming': 'Gaming enthusiast'
    }[biggest] if h[biggest + '_hours'] >= 1.5 else 'Balanced user'
