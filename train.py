from data_engine import load
from ml import train
if __name__ == '__main__':
    data, _ = load()
    b = train(data)
    print('Selected:', b['info']['winner'])
    print('Version:', b['info']['version'])
    print('Regression:', b['info']['regression'])
