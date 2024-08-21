import schedule
import time
from train_ppo import train_ppo


def real_time_train():
    """实时训练模型，并每小时保存一次"""
    train_ppo()  # 进行一次训练


def save_model_periodically():
    """定时保存模型"""
    schedule.every(1).hours.do(real_time_train)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    save_model_periodically()
